def need_char_spacing(char, last_pos_x, last_width):
    letter_spacing = 2
    if char["x0"] - (last_pos_x + last_width) - letter_spacing > 0:
        return True
    return False


def is_new_line_start(char, prev_x1):
    if char["x0"] < prev_x1:
        return True

    return False


def is_new_line_detected(top, prev_bottom, prev_height, line_height):
    return top - prev_bottom > line_height or top - prev_bottom > prev_height


def need_text_heading(char, first_char_of_line):
    if (
        first_char_of_line
        and (char.get("size", -1) >= 14 or char.get("height", -1) >= 20)
        # and is_bold_fontname(char.get("fontname", ""))
    ):
        return True
    return False


def is_bold_fontname(fontname):
    fontname = fontname.lower()
    if "bold" in fontname or "h2hdrm" in fontname:
        return True

    return False
