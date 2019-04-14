from ftplib import FTP
from io import StringIO


def download_or_open_ftp(input_list):
    """
    Download one assembly_status.txt file
    :param input_list - list including the following fields:
    1) ind: index of the current item
    2) ref_seq_ftp: path of specific ftp folder (coming after 'ftp.ncbi.nlm.nih.gov')
    3) strain_name: name of specific strain
    4) ftp_file_name: name of the ftp file to download/open
    5) ftp_file_site: site name to open the ftp connection with
    6) dest_path: destenation of the output files folder. if None then return the str retreived and don't save the file
    """
    try:
        ind = input_list[0]
        ftp_sub_folder = input_list[1]
        strain_name = input_list[2]
        ftp_file_name = input_list[3]
        ftp_site = input_list[4]
        dest_path = input_list[5]
        if ftp_sub_folder == '-':
            print(f"SKIP! ftp_sub_folder is: {ftp_sub_folder} for strain: {strain_name}, index: {ind}")
            return
        ftp = FTP(ftp_site)
        ftp.login()
        ftp.cwd(ftp_sub_folder)
        # replace "/" with "$" so it will be possible to save the file
        if dest_path is not None:
            output_file_path = dest_path + strain_name.replace("/", "$") + ".txt"
            with open(output_file_path, 'wb') as f:
                ftp.retrbinary('RETR ' + ftp_file_name, f.write)
            print(f"Downloaded file for: {strain_name}, index: {ind}")
            return
        else:
            line_reader = StringIO()
            ftp.retrlines('RETR ' + ftp_file_name, line_reader.write)
            str_to_return = line_reader.getvalue()
            line_reader.close()
            print(f"Opened file for: {strain_name}, index: {ind}")
            return [strain_name, str_to_return]
    except Exception as e:
        print(f"ERROR at downloading assembly_status for: {strain_name}, index: {ind}, message: {e}")
