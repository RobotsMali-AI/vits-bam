import argparse
from transformers import VitsModel, AutoTokenizer
from utils import VitsModelForPreTraining, VitsFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Push fine-tuned VITS model to the Hugging Face Hub.")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the saved model directory (e.g., tmp/bam_vits_exp3/)")
    parser.add_argument("--repo_id", type=str, required=True, help="Your Hugging Face repo ID (e.g., username/vits-bam-exp3)")
    parser.add_argument("--inference_only", action="store_true", help="If set, drops the discriminator and pushes only the inference weights.")

    args = parser.parse_args()

    print(f"Loading model, tokenizer, and feature extractor from '{args.save_dir}'...")

    # Choose the correct architecture based on whether you want to keep the discriminator
    if args.inference_only:
        model = VitsModel.from_pretrained(args.save_dir)
        print("Loaded VitsModel (Inference Only - Discriminator dropped).")
    else:
        model = VitsModelForPreTraining.from_pretrained(args.save_dir)
        print("Loaded VitsModelForPreTraining (Includes Discriminator for further training).")

    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    feature_extractor = VitsFeatureExtractor.from_pretrained(args.save_dir)

    print(f"Pushing to Hugging Face Hub repository: '{args.repo_id}'...")

    # Push all components
    model.push_to_hub(args.repo_id)
    tokenizer.push_to_hub(args.repo_id)
    feature_extractor.push_to_hub(args.repo_id)

    print("Success! All files have been pushed to the Hub.")

if __name__ == "__main__":
    main()
