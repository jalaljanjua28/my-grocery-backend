import json
import logging

import modules.core as core


def _get_inventory_items():
    content = core.get_data_from_json("ItemsList", "master_nonexpired")
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    if isinstance(content, str):
        content = json.loads(content)
    elif isinstance(content, dict):
        content = content
    else:
        raise TypeError("Unexpected content type returned by get_data_from_json")

    if not isinstance(content, dict) or "Food" not in content:
        raise ValueError("Invalid data format received from storage.")

    return [item for item in content["Food"] if item.get("Name") != "TestFNE"]


def _call_openai(prompt, *, max_tokens=1000, temperature=0.6):
    if not core.openai_client:
        return "OpenAI client is unavailable."

    try:
        response = core.openai_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].text.strip()
    except Exception as exc:
        logging.exception("OpenAI request failed")
        return f"Unable to generate response: {exc}"


def _save_prompt_output(folder_name, file_name, payload):
    core.save_data_to_cloud_storage(folder_name, file_name, payload)
