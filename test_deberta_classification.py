"""
Test DeBERTa's ability to classify claims vs premises
"""

from app.argmining.implementations.encoder_model_loader import NonTrainedEncoderModelLoader

def test_deberta_classification():
    print("Testing DeBERTa claim/premise classification...")
    print("-" * 60)
    
    # Initialize the model
    model = NonTrainedEncoderModelLoader(
        model_paths={
            "type_model_path": "mrkk11/deberta-stance/deberta-type-checkpoints",
            "stance_model_path": "mrkk11/deberta-stance/deberta-stance-checkpoints"
        }
    )
    
    # Test sentences - mix of likely claims and premises
    test_sentences = [
        "Climate change is the biggest threat to humanity.",  # Likely claim
        "Global temperatures have risen by 1.1 degrees Celsius since 1880.",  # Likely premise
        "We must act now to save the planet.",  # Likely claim
        "Scientists have measured CO2 levels at 420 ppm.",  # Likely premise
        "Electric vehicles are better than gas cars.",  # Likely claim
        "EVs produce zero emissions at the point of use.",  # Likely premise
    ]
    
    print("\nClassifying sentences:")
    print()
    
    for sentence in test_sentences:
        label, confidence = model._classify_adu_type(sentence)
        print(f"Text: {sentence}")
        print(f"  -> Classification: {label} (confidence: {confidence:.2%})")
        print()
    
    # Test the full classify_adus method
    print("-" * 60)
    print("Testing full text classification:")
    
    full_text = "Climate change is real. The data shows temperatures are rising. We need to take action immediately."
    result = model.classify_adus(full_text)
    
    print(f"\nText: {full_text}")
    print(f"\nClaims found: {len(result.claims)}")
    for claim in result.claims:
        print(f"  - {claim.text}")
    
    print(f"\nPremises found: {len(result.premises)}")
    for premise in result.premises:
        print(f"  - {premise.text}")

if __name__ == "__main__":
    test_deberta_classification()