from bs4 import BeautifulSoup
import xlsxwriter


#open text file in read mode
text_file = open("SVD.html", "r")
#read whole file to a string
data = text_file.read()

soup = BeautifulSoup(data, 'html.parser')

workbook = xlsxwriter.Workbook('SVD.xlsx')
worksheet = workbook.add_worksheet()

#Identification numbers of the recordings
row = 0
col = 0
for a in soup.find_all("td", "sessionidgrey"):
    worksheet.write(row, col, a.string)
    row += 1
    print(row)

#Healthy
row = 0
col = 0
for b in soup.find_all("td", "sessiontypegrey"):
    worksheet.write(row, col+1, b.string)
    row += 1

#Date of recording
row = 0
col = 0
for c in soup.find_all("td", "sessiondategrey"):
    worksheet.write(row, col+2, c.string)
    row += 1

#Identification numbers of the speakers
row = 0
col = 0
for d in soup.find_all("td", "talkeridgrey"):
    worksheet.write(row, col+3, d.string)
    row += 1

#Sex
row = 0
col = 0
for f in soup.find_all("td", "talkersexgrey"):
    worksheet.write(row, col+4, f.string)
    row += 1

#Age of the speaker at the time of the recording
row = 0
col = 0
for g in soup.find_all("td", "talkeragegrey"):
    worksheet.write(row, col+5, g.string)
    row += 1

#Pathologies
row = 0
col = 0
for h in soup.find_all("td", "pathologiesgrey"):
    worksheet.write(row, col+6, h.string)
    row += 1

#Remarks w.r.t. diagnosis
row = 0
col = 0
for i in soup.find_all("td", "diagnosisgrey"):
    worksheet.write(row, col+7, i.string)
    row += 1

#Comments
row = 0
col = 0
for j in soup.find_all("td", "commentgrey"):
    worksheet.write(row, col+8, j.string)
    row += 1

workbook.close()

