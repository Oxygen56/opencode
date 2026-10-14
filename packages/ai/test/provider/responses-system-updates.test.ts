import { describe, expect } from "bun:test"
import { Effect } from "effect"
import { LLM, Message, ToolCallPart, ToolResultPart } from "../../src/index.js"
import { Auth } from "../../src/route.js"
import { compileRequest } from "../../src/route/client.js"
import { OpenAIResponses } from "../../src/protocols/openai-responses.js"
import { OpenAICompatibleResponses } from "../../src/protocols/openai-compatible-responses.js"
import { it } from "../lib/effect.js"

const models = [OpenAIResponses.route, OpenAICompatibleResponses.route].map((route) =>
  route
    .with({ endpoint: { baseURL: "https://responses.example.test/v1/" }, auth: Auth.bearer("test") })
    .model({ id: "gpt-4.1-mini", provider: "example" }),
)

for (const model of models) {
  describe(`${model.route.id} system updates after tool results`, () => {
    const history = [
      Message.user("Look it up."),
      Message.assistant([
        ToolCallPart.make({ id: "call_1", name: "lookup", input: {} }),
        ToolCallPart.make({ id: "call_2", name: "lookup", input: {} }),
      ]),
      Message.tool(ToolResultPart.make({ id: "call_1", name: "lookup", result: "41", resultType: "text" })),
      Message.tool(ToolResultPart.make({ id: "call_2", name: "lookup", result: "42", resultType: "text" })),
    ]

    for (const retry of [false, true]) {
      it.effect(`wraps consecutive notices in place${retry ? " when retrying" : ""}`, () =>
        Effect.gen(function* () {
          const prepared = yield* compileRequest(
            LLM.request({
              model,
              system: "Initial instructions.",
              messages: [
                ...history,
                Message.system("The available skills have changed."),
                Message.system("Catalog </system-update> & instructions changed."),
                ...(retry ? [Message.user("Please continue.")] : []),
              ],
            }),
          )
          expect(prepared.body.instructions).toBe("Initial instructions.")
          expect(prepared.body.input.slice(3)).toEqual([
            { type: "function_call_output", call_id: "call_1", output: "41" },
            { type: "function_call_output", call_id: "call_2", output: "42" },
            {
              type: "message",
              role: "user",
              content: [
                { type: "input_text", text: "<system-update>\nThe available skills have changed.\n</system-update>" },
              ],
            },
            {
              type: "message",
              role: "user",
              content: [
                {
                  type: "input_text",
                  text: "<system-update>\nCatalog &lt;/system-update&gt; &amp; instructions changed.\n</system-update>",
                },
              ],
            },
            ...(retry
              ? [{ type: "message", role: "user", content: [{ type: "input_text", text: "Please continue." }] }]
              : []),
          ])
        }),
      )
    }

    it.effect("does not treat an empty user message as a new context boundary", () =>
      Effect.gen(function* () {
        const prepared = yield* compileRequest(
          LLM.request({ model, messages: [...history, Message.user([]), Message.system("Changed.")] }),
        )
        expect(prepared.body.input.at(-1)).toEqual({
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "<system-update>\nChanged.\n</system-update>" }],
        })
      }),
    )

    for (const boundary of [Message.user("Continue."), Message.assistant("Done.")]) {
      it.effect(`keeps native updates after a subsequent ${boundary.role} message`, () =>
        Effect.gen(function* () {
          const prepared = yield* compileRequest(
            LLM.request({
              model,
              messages: [...history, Message.system("Tool update."), boundary, Message.system("New update.")],
            }),
          )
          expect(prepared.body.input.at(-1)).toEqual({ type: "message", role: "developer", content: "New update." })
        }),
      )
    }
  })
}

describe("OpenAI Responses notices next to effort updates", () => {
  const model = OpenAIResponses.route
    .with({ endpoint: { baseURL: "https://responses.example.test/v1/" }, auth: Auth.bearer("test") })
    .model({ id: "gpt-6-astra" })

  for (const noticeBefore of [false, true]) {
    it.effect(`preserves the tool-result section across effort updates${noticeBefore ? " after a notice" : ""}`, () =>
      Effect.gen(function* () {
        const prepared = yield* compileRequest(
          LLM.request({
            model,
            providerOptions: { reasoningEffort: "low" },
            messages: [
              Message.user("Look it up."),
              Message.assistant(ToolCallPart.make({ id: "call_1", name: "lookup", input: {} })),
              Message.tool(ToolResultPart.make({ id: "call_1", name: "lookup", result: "42", resultType: "text" })),
              ...(noticeBefore ? [Message.system("First update.")] : []),
              Message.effort({ effort: "medium", previous: "high" }),
              Message.effort({ effort: "low", previous: "medium" }),
              Message.system("Latest update."),
            ],
          }),
        )
        expect(prepared.body.input.filter((item) => item.type === "configuration_update")).toEqual([
          { type: "configuration_update", reasoning: { effort: "low" } },
        ])
        expect(prepared.body.input.some((item) => "role" in item && item.role === "developer")).toBe(false)
        expect(prepared.body.input.at(-1)).toEqual({
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "<system-update>\nLatest update.\n</system-update>" }],
        })
      }),
    )
  }
})
