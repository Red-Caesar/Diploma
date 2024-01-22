import openai

OPENAI_VER_MAJ = int(openai.__version__.split('.', maxsplit=1)[0])

if OPENAI_VER_MAJ >= 1:
    from openai import APIError, AuthenticationError, APIConnectionError
else:
    from openai.error import APIError, AuthenticationError, APIConnectionError

def run_chat_completion(
    model_name,
    messages,
    token,
    endpoint,
    max_tokens=300,
    n=1,
    stream=False,
    stop=None,
    temperature=0.0,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0
):
    openai.api_key = token
    if OPENAI_VER_MAJ >= 1:
        client = openai.OpenAI(
            api_key=token,
            base_url = endpoint + "/v1",
        )
        chat_completions = client.chat.completions
    else:
        openai.api_base = endpoint + "/v1"
        chat_completions = openai.ChatCompletion

    completion = chat_completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        stream=stream,
        n=n,
        stop=stop,
        top_p=top_p,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

    if OPENAI_VER_MAJ >= 1 and not stream:
        return completion.model_dump(exclude_unset=True)
    return completion
    