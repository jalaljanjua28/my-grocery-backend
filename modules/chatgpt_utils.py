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


_ERROR_MARKER = "Unable to generate response due to an internal error."


def filter_error_entries(entries):
    """Strip any list items that contain the OpenAI error string (stale bad data)."""
    if not isinstance(entries, list):
        return entries
    return [
        entry for entry in entries
        if not any(
            isinstance(v, str) and _ERROR_MARKER in v
            for v in (entry.values() if isinstance(entry, dict) else [entry])
        )
    ]


def _call_openai(prompt, *, max_tokens=1000, temperature=0.6):
    if not core.openai_client:
        raise RuntimeError(
            "OpenAI client is unavailable. Configure OPENAI_API_KEY and restart the server."
        )

    try:
        response = core.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logging.exception("OpenAI request failed")
        # Let the route return an error status. Previously this was returned as
        # successful generated text, saved to storage, and then filtered out by
        # the JSON endpoint, which made the UI look as if nothing was generated.
        raise RuntimeError("OpenAI response generation failed") from exc


def save_prompt_output(folder_name, file_name, payload):
    core.save_data_to_cloud_storage(folder_name, file_name, payload)
