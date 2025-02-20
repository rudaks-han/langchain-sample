import os

from pdfplumber.page import Page


def extract_all_images(page: Page, image_append_to_page: list, image_save_dir: str):
    all_images = []

    for image_idx, image in enumerate(page.images):
        image_attrs = {
            "pos_x": image["x0"],
            "pos_y": image["top"],
            "width": image["width"],
            "height": image["height"],
        }

        x0, top, x1, bottom = (
            image["x0"],
            image["top"],
            image["x1"],
            image["bottom"],
        )

        background_image_margin = 0.1

        if (
            x1 + background_image_margin >= page.width
            or bottom + background_image_margin >= page.height
        ):
            # print("[image] 페이지 범위 벗어남")
            continue
        else:
            image_uid = f"{page.page_number}_{x0}_{top}_{x1}_{bottom}"
            if image_uid not in image_append_to_page:
                image_data = crop_image_data(page, image, x0, top, x1, bottom)
                filename = f"page_{page.page_number}_img_{image_idx}.png"
                img_save_path = os.path.join(
                    image_save_dir,
                    filename,
                )
                image_data.save(img_save_path, format="PNG")
                image_attrs["img_path"] = image_save_dir
                image_attrs["img_name"] = filename
                all_images.append(image_attrs)

                image_append_to_page.append(image_uid)

    return all_images


def extract_valid_images(all_images, top):
    results = []
    for i in range(len(all_images) - 1, -1, -1):
        if all_images[i]["pos_y"] < top:
            results.append(all_images[i])
            del all_images[i]

    return results


def crop_image_data(page: Page, image, x0, top, x1, bottom):
    if x0 < 0:
        x0 = 0
    if top < 0:
        top = 0

    return page.crop((x0, top, x1, bottom)).to_image(width=image["width"])
