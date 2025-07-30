# Text to Text Abstract

This project demonstrates the use of a pre-trained language model, **FLAN-T5 Small** by Google, to generate short inspirational quotes based on user-defined topics. 

Utilizing the Hugging Face `transformers` library along with `sentencepiece`, the script loads the `google/flan-t5-small` model and defines a simple quote generation function.

The `generate_quote()` function:
- Constructs a prompt like *"Write an inspirational quote about: [topic]"*,
- Tokenizes the input text,
- Generates the output using beam search,
- Decodes and returns the generated quote.

### Example Topics:
- Confidence  
- Self-love  
- Kindness  

This code showcases how language models can be leveraged for **creative writing and NLP-driven content generation**, making it useful for applications in mental wellness, social media, and educational tools.


# Text to image  Abstract

This project utilizes the `StableDiffusionPipeline` from the Hugging Face `diffusers` library to generate high-quality AI-generated images based on natural language prompts. The script loads the pre-trained model `runwayml/stable-diffusion-v1-5` and leverages GPU acceleration (CUDA) to enhance performance during inference.

The key steps involved include:
- Loading the Stable Diffusion model with half-precision (`float16`) for memory efficiency,
- Moving the model to the GPU (`cuda`),
- Generating an image using a descriptive text prompt (e.g., *"lord shiva"*),
- Saving and displaying the resulting image.

### Key Features:
- **Text-to-Image Synthesis**: Converts a simple prompt into a vivid, detailed image.
- **Hardware Acceleration**: Uses PyTorch's `autocast` for mixed-precision computation on CUDA.
- **Output Handling**: Saves and displays the generated image using PIL.

This implementation highlights how diffusion models can be employed for **creative image generation**, ideal for art, storytelling, concept design, and more.
