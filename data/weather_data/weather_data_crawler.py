import re
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

def crawl(month,year):
    weather_lst = []
    url = 'https://world-weather.info/forecast/usa/new_york/{}-{}/'.format(month,year)
    base_html = requests.get(url, headers=headers)    
    soup = BeautifulSoup(base_html.content,"html.parser")
    #print(soup)
    weather_block = soup.find_all('a', {'href': re.compile('//world-weather\.info/forecast/usa/new_york/[0-9]{2}-.+')})
    for i in range(len(weather_block)):
        today = weather_block[i]
        weather_info = {}
        weather_info['date'] = '{}-{}-{:02d}'.format(year,month,i+1)
        weather_info['weather'] = today.find('i')['title']
        weather_info['high_temp'] = today.find('span').text
        weather_info['low_temp'] = today.find('p').text
        weather_lst.append(weather_info)
    return weather_lst

if __name__ == '__main__':
    data = crawl('march','2020')+ crawl('april','2020') + crawl('may','2020') + crawl('june','2020') + crawl('july','2020') + crawl('august','2020') + \
       crawl('september', '2020') + crawl('october', '2020') + crawl('november', '2020') + crawl('december', '2020') + \
       crawl('january', '2021') + crawl('february', '2021') + crawl('march', '2021') + \
       crawl('april', '2021') + crawl('may', '2021') + crawl('june', '2021') + crawl('july', '2021') + \
       crawl('august', '2021') + crawl('september', '2021') + crawl('october', '2021') + crawl('november', '2021') + \
       crawl('december', '2021') + crawl('january', '2022') + crawl('february', '2022') + crawl('march', '2022') + \
       crawl('april', '2022') + crawl('may', '2022') + crawl('june', '2022') + crawl('july', '2022') + \
       crawl('august', '2022') + crawl('september', '2022') + crawl('october', '2022') + crawl('november', '2022') + \
       crawl('december', '2022') + crawl('january', '2023') + crawl('february', '2023') + crawl('march', '2023')
    for d in data:
        old_date = datetime.strptime(d['date'], '%Y-%B-%d')
        new_date = datetime.strftime(old_date, '%Y-%m-%d')
        d['date'] = new_date
    with open('weather_data.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'weather', 'high_temp', 'low_temp'])
        for item in data:
            writer.writerow([item['date'], item['weather'], item['high_temp'], item['low_temp']])
