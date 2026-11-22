from my_langsmith.config import ls_client, groq_client

def helpfulness(inputs:dict, outputs:dict, reference_outputs:dict) -> float:
    """A function that scores the helpfulness of the agent"""
    question = inputs.get('question')
    if question is None and 'inputs' in inputs:
        question = inputs['inputs'].get('question')
    response = outputs.get("response")
    if response is None and "output" in outputs:
        resposne = outputs['output'].get("response")
    answer = reference_outputs.get("answer")
    if answer is None and "output" in reference_outputs:
        answer = reference_outputs["output"].get("answer")

    user_content = f"""You will score the helpfulness of the given answer on a scale from 1-5.
    you will be given the question, the actual answer, and the predicted answer. Respond with only one number: the helpfulness score.
    Question: {question}.
    Actual answer: {answer}
    Predicted answer: {response}
    Score: 
    """
    score = groq_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": user_content
        }
    ]
).choices[0].message.content.strip()
    try:
        return float(score)
    except ValueError:
        return 0.0

    
    


    
    
