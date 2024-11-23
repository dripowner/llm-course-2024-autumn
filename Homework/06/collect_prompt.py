def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    prompt = (
        f"The following are multiple choice questions (with answers) about {sample['subject']}.\n"
        f"{sample['question']}\n"
        f"A. {sample['choices'][0]}\n"
        f"B. {sample['choices'][1]}\n"
        f"C. {sample['choices'][2]}\n"
        f"D. {sample['choices'][3]}\n"
        f"Answer:"
    )

    return prompt


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    if add_full_example:
        prompt = "\n\n".join([create_prompt(example) 
                            + f" {example['answer']}. {example['choices'][int(example['answer'])]}" 
                            for example in examples])
    else:
        prompt = "\n\n".join([create_prompt(example) 
                            + f" {answer_map[example['answer']]}" 
                            for example in examples])
        
    prompt += f"\n\n{create_prompt(sample)}"

    print(prompt)

    return prompt