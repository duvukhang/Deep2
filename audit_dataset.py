import argparse
import logging

from utils.dataset_audit import audit_yolo_dataset, write_audit_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit dataset YOLO de phat hien overfitting/leakage/label loi."
    )
    parser.add_argument("--data", default="configs/data.yaml", help="Duong dan data.yaml cua YOLO.")
    parser.add_argument("--output_json", default="dataset_report.json", help="File JSON report.")
    parser.add_argument("--output_txt", default="dataset_report.txt", help="File TXT report.")
    parser.add_argument(
        "--no_phash",
        action="store_true",
        help="Tat perceptual hash neu dataset qua lon.",
    )
    parser.add_argument(
        "--max_phash_images",
        type=int,
        default=3000,
        help="So anh toi da de tinh near-duplicate perceptual hash.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logging.info("Dang audit dataset: %s", args.data)
    report = audit_yolo_dataset(
        args.data,
        compute_phash=not args.no_phash,
        max_phash_images=args.max_phash_images,
    )
    write_audit_report(report, args.output_json, args.output_txt)

    logging.info("Da luu report: %s va %s", args.output_json, args.output_txt)
    if report["warnings"]:
        logging.warning("Phat hien %d canh bao. Xem dataset_report.txt de sua.", len(report["warnings"]))
    else:
        logging.info("Khong phat hien canh bao lon.")


if __name__ == "__main__":
    main()
