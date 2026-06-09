const { OpenAI } = require('openai');
require('dotenv').config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const executeInference = async (promptInput) => {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o", // Core decision model
      messages: [
        { 
          role: "system", 
          content: "You are the Ghost Agentic Generative Design Brain. Output strict clean JSON structures mapping physical dimension metrics, material specs, and price parameters for 3D products." 
        },
        { role: "user", content: promptInput }
      ],
      response_format: { type: "json_object" }
    });

    return JSON.parse(response.choices[0].message.content);
  } catch (error) {
    console.error("AI Node Processing failure:", error.message);
    return { error: true, message: "Inference calculation timeout encountered." };
  }
};

// Immediate testing script execution handler
if (require.main === module) {
  executeInference("Generate a sustainable structural container shell layout.")
    .then(data => console.log("🤖 Design Core Response Matrix:", data));
}

module.exports = { executeInference };