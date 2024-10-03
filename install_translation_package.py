import argostranslate.package
import argostranslate.translate
import os

# Function to download and install the English to Portuguese translation package
def install_translation_package(from_lang='en', to_lang='pt'):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Find the correct package for English to Portuguese translation
    package_to_install = next(
        filter(lambda x: x.from_code == from_lang and x.to_code == to_lang, available_packages),
        None
    )

    if package_to_install:
        # Download the package
        download_path = package_to_install.download()
        
        # Install the package from the downloaded path
        if download_path:
            argostranslate.package.install_from_path(download_path)
            print(f"Successfully installed the {from_lang} to {to_lang} translation package.")
        else:
            print(f"Failed to download {from_lang} to {to_lang} package.")
    else:
        print(f"No package found for translating from {from_lang} to {to_lang}.")

# Call the function to install the translation package
install_translation_package('en', 'pt')