from PhysicalRSI_baselines.robodojo.code_policy_demo import describe


def test_code_policy_demo_is_layout_independent_and_validated():
    artifact = describe()
    assert artifact["schema"] == "physicalrsi.robodojo.code-policy-demo/v1"
    assert artifact["layout_access"] is False
    assert artifact["source_sha256"]
