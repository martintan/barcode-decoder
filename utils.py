import random
import string


def generate_random_number() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def generate_random_text(word_count, max_word_length=10):
    return " ".join(
        generate_random_word(random.randint(3, max_word_length))
        for _ in range(word_count)
    )


def generate_random_word(length):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def add_text_to_image(draw, text, position, font, fill="black"):
    draw.text(position, text, font=font, fill=fill)
