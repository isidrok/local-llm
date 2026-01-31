import "dotenv/config";
import * as z from "zod";
import { createAgent, tool } from "langchain";
import { ChatOpenAI } from "@langchain/openai";

const getWeather = tool(
  ({ city }: { city: string }) => `It's always sunny in ${city}!`,
  {
    name: "get_weather",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string(),
    }),
  },
);

export const model = new ChatOpenAI({
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  modelName: process.env.MODEL,
  apiKey: process.env.API_KEY,
  reasoning: {
    effort: "medium",
    summary: "concise",
  },
});

export const agent = createAgent({
  model,
  tools: [getWeather],
});

async function main() {
  const stream = await agent.stream(
    {
      messages: [
        {
          role: "user",
          content:
            "what is the weather in the capital of the country whose name rhymes with pain",
        },
      ],
    },
    { streamMode: "updates" },
  );

  for await (const chunk of stream) {
    console.log(JSON.stringify(chunk, null, 2));
  }
}

main();
