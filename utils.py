import datetime


def get_file_name(prefix, ext):
    time = datetime.datetime.now()
    year = str(time.year) if time.year > 9 else '0' + str(time.year)
    month = str(time.month) if time.month > 9 else '0' + str(time.month)
    day = str(time.day) if time.day > 9 else '0' + str(time.day)
    hour = str(time.hour) if time.hour > 9 else '0' + str(time.hour)
    minute = str(time.minute) if time.minute > 9 else '0' + str(time.minute)
    file_name = "{}_{}_{}_{}_{}{}.{}".format(prefix, year, month, day, hour, minute, ext)
    return file_name
