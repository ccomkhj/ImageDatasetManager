import sys
import json


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same


def Repeat(x):
    _size = len(x)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if x[i] == x[j] and x[i] not in repeated:
                repeated.append(x[i])
    return repeated


def testt(A):
    aa = []
    for i in A:
        aa.append(i["id"])
    print("MAX {}".format(max(aa)))
    print("MIN {}".format(min(aa)))
    return Repeat(aa)


def combine(tt1, tt2, output_file):
    """
    Combine two COCO annotated files and save them into a new file.

    :param tt1: Path to the first COCO file
    :param tt2: Path to the second COCO file
    :param output_file: Path for the output JSON file
    """
    # Load JSON files
    with open(tt1) as json_file:
        d1 = json.load(json_file)
    with open(tt2) as json_file:
        d2 = json.load(json_file)

    # Check for duplicate filenames between file1 and file2
    filenames1 = [img["file_name"] for img in d1["images"]]
    filenames2 = [img["file_name"] for img in d2["images"]]
    duplicates = set(filenames1).intersection(set(filenames2))
    if duplicates:
        print(
            f"Found {len(duplicates)} duplicate filename(s). Removing them from the second file."
        )
        # Determine the image IDs in d2 that need to be removed before filtering out the images.
        removed_image_ids = {
            img["id"] for img in d2["images"] if img["file_name"] in duplicates
        }
        # Remove duplicate images from d2
        d2["images"] = [
            img for img in d2["images"] if img["file_name"] not in duplicates
        ]
        # Remove any annotations in d2 that refer to one of the duplicate images
        d2["annotations"] = [
            ann for ann in d2["annotations"] if ann["image_id"] not in removed_image_ids
        ]

    # Check if both files have the same categories (same names with same ids)
    d1_categories = {c["name"]: c["id"] for c in d1["categories"]}
    d2_categories = {c["name"]: c["id"] for c in d2["categories"]}

    for c in d1_categories:
        if c in d2_categories:
            if d1_categories[c] != d2_categories[c]:
                raise AssertionError(
                    "Category name: {}, id: {} in file 1 and {} in file 2".format(
                        c, d1_categories[c], d2_categories[c]
                    )
                )
        else:
            raise AssertionError(
                "Category name: {} in file 1 does not exist in file 2".format(c)
            )

    for c in d2_categories:
        if c in d1_categories:
            if d1_categories[c] != d2_categories[c]:
                raise AssertionError(
                    "Category name: {}, id: {} in file 1 and {} in file 2".format(
                        c, d1_categories[c], d2_categories[c]
                    )
                )
        else:
            raise AssertionError(
                "Category name: {} in file 2 does not exist in file 1".format(c)
            )

    # Reset image IDs for both files to avoid conflicts
    b1 = {img["id"]: idx for idx, img in enumerate(d1["images"])}
    # Start numbering d2 images after the last id in d1
    b2 = {img["id"]: idx + max(b1.values()) + 1 for idx, img in enumerate(d2["images"])}

    for img in d1["images"]:
        img["id"] = b1[img["id"]]
    for img in d2["images"]:
        img["id"] = b2[img["id"]]

    # Reset annotations id and update image_ids for annotations
    b3 = {ann["id"]: idx for idx, ann in enumerate(d1["annotations"])}
    b4 = {
        ann["id"]: idx + max(b3.values()) + 1
        for idx, ann in enumerate(d2["annotations"])
    }

    for ann in d1["annotations"]:
        ann["id"] = b3[ann["id"]]
        ann["image_id"] = b1[ann["image_id"]]
    for ann in d2["annotations"]:
        ann["id"] = b4[ann["id"]]
        ann["image_id"] = b2[ann["image_id"]]

    # Combine the two datasets into one
    combined = d1.copy()
    combined["images"].extend(d2["images"])
    combined["annotations"].extend(d2["annotations"])
    combined["categories"] = d2["categories"]

    # Save the combined dataset to the output file
    with open(output_file, "w") as f:
        json.dump(combined, f)


if __name__ == "__main__":
    if "-h" in sys.argv or len(sys.argv) != 4:
        print(
            "\nUsage: python {} <path_to_file_1> <path_to_file_2> <output_file>\n\n"
            "Requirements:\n"
            "1- There shouldn't be duplicate image_names in the two files\n"
            "2- The two files should have the same categories (same names and ids)\n".format(
                sys.argv[0]
            )
        )
        exit(1)

    combine(sys.argv[1], sys.argv[2], sys.argv[3])
    print(
        "\n\nSuccessfully merged the two files ({} , {}) into {}".format(
            sys.argv[1], sys.argv[2], sys.argv[3]
        )
    )
