from pathlib import Path

import pytest
import yaml

from qwen_cua.deploy import build_command, load_profile

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_all_model_profiles_build_complete_vllm_commands(config_path):
    profile = load_profile(config_path)
    command = build_command(profile, "/opt/vllm/bin/vllm")

    assert command[:3] == ["/opt/vllm/bin/vllm", "serve", profile["model"]["id"]]
    assert command[command.index("--revision") + 1] == profile["model"]["revision"]
    assert command[command.index("--served-model-name") + 1] == profile["model"]["served_name"]
    assert command[command.index("--max-model-len") + 1] == str(profile["serving"]["max_model_len"])
    template_kwargs = command[command.index("--default-chat-template-kwargs") + 1]
    assert yaml.safe_load(template_kwargs) == {
        "enable_thinking": profile["inference"]["enable_thinking"]
    }
    image_limit = command[command.index("--limit-mm-per-prompt") + 1]
    assert yaml.safe_load(image_limit) == {"image": profile["inference"]["image_max"]}


def test_profile_rejects_fields_the_deploy_script_would_ignore(tmp_path):
    raw = yaml.safe_load((CONFIG_DIR / "qwen3.5_9b_nothink.yaml").read_text())
    raw["serving"]["mystery_flag"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="unknown: mystery_flag"):
        load_profile(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("inference", "top_p", 1.5, "inference.top_p must be in"),
        ("inference", "max_tokens", -1, "inference.max_tokens must be"),
        ("serving", "port", 70000, "serving.port must be"),
        ("serving", "data_parallel_size", "8", "must be an integer"),
        ("serving", "gpu_memory_utilization", 0, "must be in"),
    ],
)
def test_profile_rejects_invalid_values(tmp_path, section, key, value, message):
    raw = yaml.safe_load((CONFIG_DIR / "qwen3.5_9b_nothink.yaml").read_text())
    raw[section][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match=message):
        load_profile(path)
