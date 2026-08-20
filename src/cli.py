"""
Main CLI entry point for Yale Scanning QC.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def main():
    parser = argparse.ArgumentParser(
        description="Yale Scanning QC - Image Quality Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serve            Run the QC web app (scheduler included)
  run-batch        Run one QC pass over the monitored directories now
  retrain          Retrain a model from accumulated reviewer feedback
  models           List registered model versions
  activate-model   Activate a specific model version by id

  download-models  Download pre-trained YOLO models from HuggingFace (legacy)
  download         Download images from IIIF manifests
  augment          Augment images for training
  prepare          Prepare datasets for YOLO training
  train            Train YOLO26 models (legacy multi-category)
  analyze          Analyze a single image (legacy analyzer)

Examples:
  # Run web app (UI + daily scheduler)
  uv run yale-qc serve --host 0.0.0.0

  # Run one pass right now (e.g. from Task Scheduler / cron)
  uv run yale-qc run-batch

  # Retrain the binary bad-scan model from reviews
  uv run yale-qc retrain --role badscan --epochs 50
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download models command
    dm_parser = subparsers.add_parser("download-models", help="Download pre-trained YOLO models from HuggingFace")
    dm_parser.add_argument("-o", "--output-dir", type=Path, help="Directory to save models (default: models/)")
    dm_parser.add_argument("--force", action="store_true", help="Re-download even if models already exist")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download images from manifests")
    dl_parser.add_argument("-n", "--num-images", type=int, default=5000)
    dl_parser.add_argument("-r", "--max-resolution", type=int, default=1500)
    dl_parser.add_argument("-t", "--threads", type=int, default=20)
    dl_parser.add_argument("-s", "--seed", type=int, default=1337)
    dl_parser.add_argument("-m", "--manifest-file", type=Path, help="Path to manifests.json")
    dl_parser.add_argument("-o", "--output-dir", type=Path)

    # Augment command
    aug_parser = subparsers.add_parser("augment", help="Augment images for training")
    aug_parser.add_argument("-i", "--input-dir", type=Path)
    aug_parser.add_argument("-o", "--output-dir", type=Path)
    aug_parser.add_argument("-c", "--categories", nargs="+")
    aug_parser.add_argument("-n", "--num-samples", type=int, default=5)
    aug_parser.add_argument("-s", "--seed", type=int, default=1337)

    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Prepare datasets for training")
    prep_parser.add_argument("-i", "--input-dir", type=Path)
    prep_parser.add_argument("-o", "--output-dir", type=Path)
    prep_parser.add_argument("-c", "--category")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLO26 models")
    train_parser.add_argument("-c", "--category")
    train_parser.add_argument("-d", "--dataset-dir", type=Path)
    train_parser.add_argument("-m", "--model-size", choices=["n", "s", "m", "l", "x"], default="s")
    train_parser.add_argument("-e", "--epochs", type=int, default=100)
    train_parser.add_argument("-b", "--batch-size", type=int, default=16)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the QC web app")
    serve_parser.add_argument("-p", "--port", type=int, default=5000)
    serve_parser.add_argument("-d", "--debug", action="store_true")
    serve_parser.add_argument("--host", default="127.0.0.1")

    # Batch pipeline commands
    rb_parser = subparsers.add_parser("run-batch", help="Run one QC pass now")
    rb_parser.add_argument("--dirs", nargs="+", type=Path,
                           help="Override monitored directories for this pass")

    rt_parser = subparsers.add_parser("retrain", help="Retrain a model from reviewer feedback")
    rt_parser.add_argument("--role", choices=["badscan", "pagetype"], default="badscan")
    rt_parser.add_argument("-e", "--epochs", type=int, default=50)
    rt_parser.add_argument("--no-activate", action="store_true",
                           help="Register the new version without activating it")

    ts_parser = subparsers.add_parser("train-server",
                                      help="Run the companion GPU train/infer server (on the remote box)")
    ts_parser.add_argument("--host", default="0.0.0.0")
    ts_parser.add_argument("--port", type=int, default=8100)

    subparsers.add_parser("models", help="List registered model versions")

    am_parser = subparsers.add_parser("activate-model", help="Activate a model version")
    am_parser.add_argument("model_id", type=int)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single image")
    analyze_parser.add_argument("image", type=Path, help="Path to image file")
    analyze_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "download-models":
        from huggingface_hub import hf_hub_download
        from .config import MODELS_DIR, QC_CATEGORIES, HF_REPO_ID

        output_dir = args.output_dir or MODELS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        model_files = [
            config["model"]
            for config in QC_CATEGORIES.values()
            if config.get("model")
        ]

        for filename in model_files:
            dest = output_dir / filename
            if dest.exists() and not args.force:
                print(f"  - {filename} already exists, skipping (use --force to re-download)")
                continue
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=output_dir,
            )
            print(f"  ✓ {filename}")

        print(f"\nModels saved to {output_dir}")

    elif args.command == "download":
        from .downloader import download_images, DEFAULT_MANIFEST_FILE
        from .config import RAW_IMAGES_DIR

        download_images(
            num_images=args.num_images,
            max_resolution=args.max_resolution,
            num_threads=args.threads,
            seed=args.seed,
            manifest_file=args.manifest_file or DEFAULT_MANIFEST_FILE,
            output_dir=args.output_dir or RAW_IMAGES_DIR,
        )

    elif args.command == "augment":
        from .augmentation.pipeline import AugmentationPipeline
        from .config import AUGMENTED_DIR, RAW_IMAGES_DIR

        pipeline = AugmentationPipeline(seed=args.seed)
        pipeline.process_directory(
            input_dir=args.input_dir or RAW_IMAGES_DIR,
            output_dir=args.output_dir or AUGMENTED_DIR,
            categories=args.categories,
            num_samples_per_category=args.num_samples,
        )

    elif args.command == "prepare":
        from .training.prepare_data import DatasetPreparer
        from .config import AUGMENTED_DIR

        preparer = DatasetPreparer(
            augmented_dir=args.input_dir or AUGMENTED_DIR,
            output_dir=args.output_dir,
        )
        if args.category:
            from .config import QC_CATEGORIES
            config = QC_CATEGORIES[args.category]
            if config["type"] == "classification":
                preparer.prepare_classification_dataset(args.category, config["classes"])
            else:
                preparer.prepare_detection_dataset(args.category, config["classes"])
        else:
            preparer.prepare_all_datasets()

    elif args.command == "train":
        from .training.train import YOLOTrainer

        trainer = YOLOTrainer(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if args.category:
            trainer.train_category(args.category, args.dataset_dir)
        else:
            trainer.train_all()

    elif args.command == "serve":
        from .app import app
        print(f"Starting QC app at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)

    elif args.command == "run-batch":
        from . import db, pipeline, registry

        db.init_db()
        registry.seed_registry()
        dirs = [str(d.expanduser().resolve()) for d in args.dirs] if args.dirs else None
        batch_id = pipeline.run_batch(trigger="manual", directories=dirs)
        if batch_id is None:
            print("A batch is already running.")
            sys.exit(1)
        batch = db.get_batch(batch_id)
        print(f"\n{batch['label']}: {batch['total']} scanned, "
              f"{batch['n_bad']} flagged, {batch['n_clean']} clean, "
              f"{batch['n_unknown']} unknown, {batch['n_errors']} errors")

    elif args.command == "retrain":
        from . import db, registry, retrain

        db.init_db()
        registry.seed_registry()
        record = retrain.train(args.role, epochs=args.epochs,
                               auto_activate=not args.no_activate)
        if record is None:
            print("A training run is already in progress.")
            sys.exit(1)
        print(f"Registered {record['version']} at {record['path']}")

    elif args.command == "train-server":
        import sys as _sys
        from . import train_server
        _sys.argv = ["train-server", "--host", args.host, "--port", str(args.port)]
        train_server.main()

    elif args.command == "models":
        from . import db, registry

        db.init_db()
        registry.seed_registry()
        for m in db.list_models():
            active = " *ACTIVE*" if m["is_active"] else ""
            print(f"[{m['id']:3d}] {m['version']:20s} {m['created_at'][:19]}{active}")
            print(f"      {m['path']}")

    elif args.command == "activate-model":
        from . import db

        db.init_db()
        record = db.activate_model(args.model_id)
        if record is None:
            print(f"No model with id {args.model_id}")
            sys.exit(1)
        print(f"Activated {record['version']}")

    elif args.command == "analyze":
        import json
        from .qc.analyzer import QCAnalyzer

        analyzer = QCAnalyzer()
        results = analyzer.analyze_file(args.image)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"\n{'='*50}")
            print(f"QC Analysis: {args.image.name}")
            print(f"{'='*50}\n")

            for category, result in results.items():
                if category in ["image", "overall"]:
                    continue
                if isinstance(result, dict):
                    status = result.get("status", "unknown")
                    status_icon = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
                    print(f"{status_icon} {category.upper()}: {result.get('details', 'N/A')}")

            print(f"\n{'='*50}")
            overall = results.get("overall", "unknown")
            overall_icon = "✅" if overall == "pass" else "❌" if overall == "fail" else "⚠️"
            print(f"OVERALL: {overall_icon} {overall.upper()}")
            print(f"{'='*50}\n")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

