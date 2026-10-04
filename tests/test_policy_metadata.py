from PhysicalRSI_baselines.robodojo.skills.metadata import ExternalPolicyMetadata


class ExampleSkill(ExternalPolicyMetadata):
    policy_name = "example-vla"


def test_skill_is_evaluation_only():
    descriptor = ExampleSkill.descriptor()
    assert descriptor["training_available"] is False
    assert descriptor["evaluation_only"] is True
    assert not hasattr(ExampleSkill, "training_request")
