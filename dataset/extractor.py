import os
import shutil
import rarfile
import gdown # اگر gdown نصب نیست: !pip install gdown

# برای استفاده از دستورات shell مانند !unrar در Colab، نیاز به import کردن sys است
# اگرچه ! در سلول های Colab مستقیماً کار می کند، اما در توابع Python باید از os.system یا subprocess استفاده کرد.
# در این مثال، برای سادگی و کارایی در Colab، فرض می کنیم !unrar مستقیما کار می کند.
# اگر در تابعی خارج از محیط مستقیم سلول Colab به مشکل خوردید، از os.system استفاده کنید:
# os.system(f"unrar x -o+ {output_file} {extract_path}/")

class DataExtractor:
    def __init__(self, extract_base_path="extracted_data_dir"):
        """
        کلاس را مقداردهی اولیه می‌کند.
        :param extract_base_path: مسیر پایه برای استخراج فایل‌ها.
        """
        self.extract_base_path = extract_base_path
        print(f"DataExtractor مقداردهی اولیه شد. مسیر استخراج پیش فرض: {self.extract_base_path}")

    def extract_data_file(self, file_id, output_file_name="data.rar"):
        """
        فایل RAR را از Google Drive دانلود، استخراج و محتویات آن را نمایش می‌دهد.
        پوشه استخراج شده در مسیر self.extract_base_path قرار می‌گیرد.
        :param file_id: شناسه فایل Google Drive.
        :param output_file_name: نام فایل RAR دانلود شده.
        """
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path_rar = os.path.join(self.extract_base_path, output_file_name) # ذخیره rar داخل پوشه اصلی

        print(f"در حال دانلود فایل از Google Drive (ID: {file_id})...")
        try:
            # مطمئن شوید که پوشه پایه موجود است تا فایل rar داخل آن ذخیره شود
            os.makedirs(self.extract_base_path, exist_ok=True)
            gdown.download(url, output_path_rar, quiet=False)
            print(f"فایل '{output_file_name}' با موفقیت دانلود شد.")
        except Exception as e:
            print(f"خطا در دانلود فایل: {e}")
            return # خروج اگر دانلود شکست خورد

        print(f"در حال استخراج فایل RAR به: {self.extract_base_path}...")
        try:
            # استخراج فایل RAR. محتویات به extract_base_path/ استخراج می شوند.
            # توجه: دستور !unrar در محیط Colab مستقیماً کار می کند.
            # اگر این کد را در محیطی غیر از Colab یا داخل تابعی خارج از سلول مستقیم استفاده کنید،
            # باید از os.system(f"unrar x -o+ {output_path_rar} {self.extract_base_path}/") استفاده کنید.
            with rarfile.RarFile(output_path_rar, 'r') as rf:
                rf.extractall(self.extract_base_path)
            print(f"فایل '{output_file_name}' با موفقیت استخراج شد.")
        except Exception as e:
            print(f"خطا در استخراج فایل RAR: {e}")
            return # خروج اگر استخراج شکست خورد

        # نمایش محتویات پوشه استخراج شده
        print("\n--- محتویات پوشه استخراج شده ---")
        for root, dirs, files in os.walk(self.extract_base_path):
            level = root.replace(self.extract_base_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}📂 {os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f'{subindent}📄 {f}')
        print("--- پایان نمایش محتویات ---")

    def clean_extracted_data(self):
        """
        پوشه اصلی استخراج شده را به صورت کامل پاک می‌کند
        تا فضای دیسک را آزاد کند و مصرف رم (از نظر فایل‌های کش شده) را کاهش دهد.
        """
        if os.path.exists(self.extract_base_path):
            print(f"در حال پاک کردن پوشه '{self.extract_base_path}'...")
            try:
                # shutil.rmtree برای حذف کامل یک دایرکتوری و تمام محتویات آن است.
                # این روش بسیار کارآمد است و تمام فایل‌ها و پوشه‌های فرعی را پاک می‌کند.
                shutil.rmtree(self.extract_base_path)
                print(f"پوشه '{self.extract_base_path}' با موفقیت پاک شد و فضا آزاد شد.")
            except OSError as e:
                print(f"خطا در پاک کردن پوشه {self.extract_base_path}: {e}. لطفاً دسترسی‌ها را بررسی کنید.")
        else:

            print(f"پوشه '{self.extract_base_path}' وجود ندارد. چیزی برای پاک کردن نیست.")

