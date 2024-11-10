import os
import cv2


def load_images_and_labels(data_dir, set_type):
    images = []
    labels = []

    # Ścieżki do folderów obrazów i etykiet
    images_path = os.path.join(data_dir, set_type, 'images')
    labels_path = os.path.join(data_dir, set_type, 'labels')

    # Iteracja przez wszystkie pliki w folderze z obrazami
    for filename in os.listdir(images_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Ścieżka do etykiety
            label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
            label_fullpath = os.path.join(labels_path, label_filename)

            # Sprawdzenie czy istnieje odpowiadający plik etykiety
            if os.path.exists(label_fullpath):
                with open(label_fullpath, 'r') as f:
                    label_data = f.readlines()

                    # Zakładamy, że pierwsza wartość w pliku etykiety to informacja o skoliozie
                    if label_data:
                        first_value = int(label_data[0].strip().split()[0])

                        # Wczytanie obrazu tylko jeśli istnieje etykieta
                        img = cv2.imread(os.path.join(images_path, filename))
                        images.append(img)
                        labels.append(first_value)
            else:
                print(f"Brak etykiety dla obrazu {filename}, pomijam...")

    return images, labels