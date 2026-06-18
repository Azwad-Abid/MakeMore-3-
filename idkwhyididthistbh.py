if not os.path.exists('names.txt'):
    print("Downloading names.txt...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt',
        'names.txt'
    )
    print("Downloaded!\n")
