import "dotenv/config";
import * as z from "zod";
import { createAgent, tool } from "langchain";
import { ChatOpenAI } from "@langchain/openai";

const getWeather = tool(({ city }) => `It's always sunny in ${city}!`, {
  name: "get_weather",
  description: "Get the weather for a given city",
  schema: z.object({
    city: z.string(),
  }),
});

const llm = new ChatOpenAI({
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  modelName: process.env.MODEL,
  apiKey: process.env.API_KEY,
});

const agent = createAgent({
  model: llm,
  tools: [getWeather],
});

async function run() {
  const response = await agent.invoke({
    messages: [{ role: "user", content: "What's the weather in New York?" }],
  });
  console.log(response);
}

run();
