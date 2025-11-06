import argparse
import os
import sys


def is_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, IOError):
        return False


def gather_files(start_dir, exclude_dirs):
    output_path = os.path.join(start_dir, "backend.txt")
    skipped = 0
    processed = 0

    # Преобразуем имена исключаемых директорий в нижний регистр для case-insensitive сравнения
    exclude_dirs = [d.lower() for d in exclude_dirs]

    with open(output_path, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(start_dir, topdown=True):
            # Удаляем исключенные директории из списка для обхода
            dirs[:] = [
                d
                for d in dirs
                if d.lower() not in exclude_dirs
                and not d.startswith(".")
                and not d.startswith("__")
            ]

            for filename in files:
                filepath = os.path.join(root, filename)

                # Пропускаем скрытые файлы и backend.txt
                if filename.startswith(".") or filename.lower() == "backend.txt":
                    skipped += 1
                    continue

                if is_text_file(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as infile:
                            content = infile.read()

                        rel_path = os.path.relpath(filepath, start_dir)
                        outfile.write(f"Файл: {rel_path}\n")
                        outfile.write("-" * 80 + "\n")
                        outfile.write(content + "\n")
                        outfile.write("=" * 80 + "\n\n")
                        processed += 1
                    except Exception as e:
                        print(f"Ошибка при обработке {filepath}: {str(e)}")
                        skipped += 1
                else:
                    skipped += 1

    print(f"Готово! Обработано файлов: {processed}, пропущено: {skipped}")
    print(f"Результат сохранён в: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Собирает содержимое всех текстовых файлов в backend.txt"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.getcwd(),
        help="Целевая директория (по умолчанию: текущая)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        action="append",
        default=[],
        help="Директории для исключения (можно указывать несколько раз)",
    )

    args = parser.parse_args()

    # Стандартные исключения + пользовательские
    default_excludes = ["venv", "__pycache__", ".git", "node_modules", "build"]
    exclude_dirs = list(set(default_excludes + args.exclude))

    print(f"Исключаемые директории: {', '.join(exclude_dirs)}")
    gather_files(args.directory, exclude_dirs)