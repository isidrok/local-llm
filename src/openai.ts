import "dotenv/config";
import { Agent, tool, run, OpenAIChatCompletionsModel } from "@openai/agents";
import { z } from "zod";
import OpenAI from "openai";

const getWeatherTool = tool({
  name: "get_weather",
  description: "Get the weather for a given city",
  parameters: z.object({ city: z.string() }),
  async execute({ city }) {
    return `It's always sunny in ${city}!`;
  },
});

const client = new OpenAI({
  baseURL: process.env.BASE_URL,
  apiKey: process.env.API_KEY,
});

const agent = new Agent({
  name: "Weather Agent",
  instructions:
    "You are a helpful assistant that can check weather information.",
  model: new OpenAIChatCompletionsModel(client, process.env.MODEL!),
  modelSettings: {
    reasoning: {
      effort: "medium",
      summary: "concise",
    },
  },
  tools: [getWeatherTool],
});

async function main() {
  const result = await run(
    agent,
    "what is the weather in the capital of the country whose name rhymes with pain",
    {
      stream: true,
    },
  );

  for await (const event of result) {
    console.log(JSON.stringify(event, null, 2));
  }
}

main();
