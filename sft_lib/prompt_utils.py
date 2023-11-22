"""
How to prompt Llama2: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
Llama2 prompt format:
[INST] «SYS»\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n«/SYS»\n\nHi, how are you? [/INST] Good thanks! \n[INST] Can you help me with this math program? [/INST]"""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"


def get_sys_prompt_template(assistant_action: str = None):
    """
    example of assistant_action: "analyze Request for Proposal documents"
    example of sys prompt: "You are a helpful, respectful and honest assistant who can help analyze Request for Proposal documents, and always answer as helpfully as possible. If you don't know the answer to a question, please don't share false information."
    """
    _get_prompt = (
        lambda x: f"""You are a helpful, respectful and honest assistant{x}, and always answer as helpfully as possible. If you don't know the answer to a question, please don't share false information."""
    )
    if assistant_action and len(assistant_action.strip()) > 0:
        return _get_prompt(f" who can help {assistant_action.strip()}")
    return _get_prompt("")


def llama2_prompt_format(
    messages: list[dict], add_sys_prompt: bool = True, assistant_action: str = None
):
    if add_sys_prompt and messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": get_sys_prompt_template(assistant_action),
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": f"{B_SYS}{messages[0]['content']}{E_SYS}{messages[1]['content']}",
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def text2prompt(text: str, add_sys_prompt: bool = True, assistant_action: str = None):
    return llama2_prompt_format(
        messages=[{"role": "user", "content": text}],
        add_sys_prompt=add_sys_prompt,
        assistant_action=assistant_action,
    )


if __name__ == "__main__":
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
        {"role": "assistant", "content": "Hi Orange, how are you?"},
        {"role": "user", "content": "I am good, and you?"},
        {"role": "assistant", "content": "I am also good, thanks!"},
    ]

    # print(convert_openai_to_llama_format(messages))
    # print(llama2_prompt_format(messages))
    print(text2prompt("Knock knock."))
