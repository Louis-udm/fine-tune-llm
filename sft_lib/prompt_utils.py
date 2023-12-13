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
    answer_label = None
    if add_sys_prompt and messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": get_sys_prompt_template(assistant_action),
            }
        ] + messages
    if messages[0]["role"] == "system":
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
    if len(messages) % 2 == 1:
        # the last message is from user
        messages_list.append(
            f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}"
        )
    else:
        # the last message is from assistant,
        # this case is for training the model, the last message is what we want the answer from the model,
        # so we delete last EOS
        messages_list[-1] = messages_list[-1][: -len(EOS)].strip()
        answer_label = messages[-1]["content"].strip()

    return "".join(messages_list), answer_label


def text2prompt(
    text: str,
    add_sys_prompt: bool = True,
    assistant_action: str = None,
    answer_delimiter: str = "^^^^A^^^^",
):
    answer = None
    answer_idx = text.find(answer_delimiter)
    if answer_idx > -1:
        text, answer = (
            text[:answer_idx].strip(),
            text[answer_idx + len(answer_delimiter) :].strip(),
        )
    messages = [{"role": "user", "content": text}]
    if answer:
        messages.append({"role": "assistant", "content": answer})
    return llama2_prompt_format(
        messages=messages,
        add_sys_prompt=add_sys_prompt,
        assistant_action=assistant_action,
    )


if __name__ == "__main__":
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange. What's your name?"},
    ]

    # print(convert_openai_to_llama_format(messages))
    assert llama2_prompt_format(messages) == (
        """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant, and always answer as helpfully as possible. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Knock knock. [/INST] Who's there? </s><s>[INST] Orange. What's your name? [/INST]""",
        None,
    )
    assert llama2_prompt_format(messages, assistant_action="tell a story") == (
        """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant who can help tell a story, and always answer as helpfully as possible. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Knock knock. [/INST] Who's there? </s><s>[INST] Orange. What's your name? [/INST]""",
        None,
    )
    assert llama2_prompt_format(messages, add_sys_prompt=False) == (
        "<s>[INST] Knock knock. [/INST] Who's there? </s><s>[INST] Orange. What's your name? [/INST]",
        None,
    )

    # this is the text for traning the model
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange. What's your name?"},
        {
            "role": "assistant",
            "content": "Hi Orange. My name is LLM. Nice to meet you.",
        },
    ]
    assert llama2_prompt_format(messages, add_sys_prompt=False) == (
        "<s>[INST] Knock knock. [/INST] Who's there? </s><s>[INST] Orange. What's your name? [/INST] Hi Orange. My name is LLM. Nice to meet you.",
        messages[-1]["content"],
    )
    assert text2prompt("Knock knock.\n^^^^A^^^^Who's there?", add_sys_prompt=False) == (
        "<s>[INST] Knock knock. [/INST] Who's there?",
        "Who's there?",
    )
