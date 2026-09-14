import os
import random
import json


DATASET_ROOT = "/Users/huzaifa/Documents/sync"

OUTPUT_FILE = "./splits/user_split.json"

SEED = 42

random.seed(SEED)


# Find users
users = [
    name
    for name in os.listdir(DATASET_ROOT)
    if os.path.isdir(os.path.join(DATASET_ROOT, name))
]

users.sort()

print("Total users:", len(users))


# Shuffle deterministically
random.shuffle(users)


# 80 / 10 / 10
n = len(users)

train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_users = users[:train_end]
val_users = users[train_end:val_end]
test_users = users[val_end:]


# Create output directory
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# Save split
split = {
    "train": train_users,
    "validation": val_users,
    "test": test_users
}


with open(OUTPUT_FILE, "w") as f:
    json.dump(
        split,
        f,
        indent=4
    )


print("\nSaved user split to:")
print(OUTPUT_FILE)

print("\nTrain:", len(train_users))
print("Validation:", len(val_users))
print("Test:", len(test_users))