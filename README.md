# Karpathy Challenge

This project details our attempt at tackling the [Karpathy LLM Challenge](https://twitter.com/karpathy/status/1760740503614836917) using the OpenAI GenAI stack solely. For an in-depth understanding of the challenge's vision, refer to this [sample output](https://t.co/AybDNA28sC).

## Execution 

The project is dockerized and the Dockerfile is located in the Docker folder along with the system package requirements. To build and run the Docker image, follow these steps:

```bash
# Build the Docker image
docker build -t karpathy-challenge ./Docker    
# Run the Docker image
docker run -v $(pwd):/Karpathy-Challenge -p 8501:8501 karpathy-challenge
```
Post this, view the Streamlit webpage on localhost:8501 to commence interaction.

## Future Goals
1. Enhance cost-effectiveness per video execution
2. Accelerate speed with parallel calls
3. Evaluate the possibility of replacing GPT with open-source LLMs

## Points of Reference

For further technical comprehension, review our [design inspiration](https://x.com/MisbahSy/status/1763639317270786531).

We welcome feedback, contributions or innovative ideas for future enhancement.