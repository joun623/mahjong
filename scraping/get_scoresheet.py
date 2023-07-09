from bs4 import BeautifulSoup
import requests

url_list = ["secret1", "secret2"]
source_url = "secret3"

def get_haihu():
    NG_URL_THRESHOLD = 5 
    ng_url_num = 0 
    haihu_url = []

    for i in range(1, 1000):
        url = source_url + str(i) + ".html" 
        html = requests.get(url, "html5")

        if html.status_code != requests.codes.ok:
            ng_url_num += 1
            print(url + "は見つかりませんでした")
            if ng_url_num > NG_URL_THRESHOLD:
                print("以降見つからなさそうなので、取得を終了します")
                output_file(haihu_url)
                return 
        else:
            ng_url_num = 0



        soup = BeautifulSoup(html.text, "lxml")
        # soup = BeautifulSoup(html.text, "htmp.parser")

        title = soup.title.string
        # if "secret4" not in title:
        if "secret5" not in title:
            print("no_read: " + title)
            continue
        else :
            print(title + "を読み込みます.")

        a_tag = soup.find_all("a")

        for a in a_tag:
            href_str = a.get('href')
            if "http://tenhou.net/0/?log" in href_str:
                haihu_url.append(href_str)

def output_file(haihu_url):
    with open("haihu_url.txt", "w") as f:
        for url in haihu_url:
            f.write(url + "\n") 


if __name__ == "__main__":
    get_haihu()
