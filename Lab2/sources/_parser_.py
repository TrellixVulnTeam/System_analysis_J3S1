class Parser:

    def parse(file_path):
        data_file = []
        with open(file_path, 'r') as read_file:
            for line in read_file:
                my_string = line.strip('\n ')
                data_file.append(my_string.split(';'))
            return data_file
