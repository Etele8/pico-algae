def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)