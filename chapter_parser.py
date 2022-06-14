import os
import re

"""
This file contains functions to separate a .txt book into chapters
"""


class ChapterParser:
    def __init__(self, filename):
        self.filename = filename
        self.contents = self.get_contents()
        self.lines = self.get_lines()
        self.end_location = self.get_end_location()

        self.parse_error = False
        self.chapter_locations = self.get_chapter_locations()

        if self.parse_error is False:
            self.remove_table_of_contents()
            self.chapter_texts = self.get_text_between_chapters()
            self.write_chapters()

    def get_contents(self):
        """
        Read the file and store its contents
        :return: str
        """
        with open(self.filename, "r", encoding="utf-8-sig") as f:
            contents = f.read()
        return contents

    def get_lines(self):
        """
        Split the contents of the human_books into
        :return: list <str>
        """
        return self.contents.split('\n')

    def get_chapter_locations(self):
        """
        Find the line number in the book where each chapter is located
        :return: list <int>
        """
        # Form 1: Chapter I, Chapter 1, Chapter the First, CHAPTER 1
        # Ways of enumerating chapters, e.g.
        arabic_numerals = '\d+'
        roman_numerals = '(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})'
        tens_in_words = ['twenty', 'thirty', 'forty', 'fifty', 'sixty',
                         'seventy', 'eighty', 'ninety']
        digits_in_words = ['one', 'two', 'three', 'four', 'five', 'six',
                           'seven', 'eight', 'nine', 'ten', 'eleven',
                           'twelve', 'thirteen', 'fourteen', 'fifteen',
                           'sixteen', 'seventeen', 'eighteen', 'nineteen'] + tens_in_words
        number_in_words_pattern = '(' + '|'.join(digits_in_words) + ')'
        ordinal_number_words_in_tens = ['twentieth', 'thirtieth', 'fortieth', 'fiftieth',
                                        'sixtieth', 'seventieth', 'eightieth', 'ninetieth'] + \
                                       tens_in_words
        ordinal_number_words = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth',
                                'seventh', 'eighth', 'ninth', 'twelfth', 'last'] + \
                               [numberWord + 'th' for numberWord in digits_in_words] + ordinal_number_words_in_tens
        ordinals_pattern = '(the )?(' + '|'.join(ordinal_number_words) + ')'
        enumerators = [arabic_numerals, roman_numerals, number_in_words_pattern, ordinals_pattern]
        form1 = 'chapter ' + '(' + '|'.join(enumerators) + ')'

        # Form 2: II. The Mail
        separators = '(\. | )'
        case = '[A-Z][a-z]'
        form2 = roman_numerals + separators + case

        # Form 3: II. THE OPEN ROAD
        enumerators = roman_numerals
        separators = '(\. )'
        case = '[A-Z][A-Z]'
        form3 = enumerators + separators + case

        # Form 4: a number on its own, e.g. 8, VIII
        arabic_numerals = '^\d+\.?$'
        roman_numerals = '(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?$'
        enumerators = [arabic_numerals, roman_numerals]
        enumerators = '(' + '|'.join(enumerators) + ')'
        form4 = enumerators

        pat = re.compile(form1, re.IGNORECASE)
        # This one is case-sensitive.
        pat2 = re.compile('(%s|%s|%s)' % (form2, form3, form4))

        headings = []  # Track line numbers of where a heading appears
        for i, line in enumerate(self.lines):
            if pat.match(line) is not None:
                headings.append(i)
            if pat2.match(line) is not None:
                headings.append(i)

        if len(headings) < 3:  # Error in parsing chapters
            print("Error in parsing " + self.filename)
            self.parse_error = True

        # Treat the end location as a heading.
        headings.append(self.end_location)

        return headings

    def remove_table_of_contents(self):
        """
        Find and delete line numbers that actually belongs to table of contents from heading_locations
        :return: list <int>
        """
        toc_locations = []

        pairs = zip(self.chapter_locations, self.chapter_locations[1:])
        for ch1, ch2 in pairs:  # Compare adjacent heading locations
            distance = ch2 - ch1
            if distance < 4:
                if ch1 not in toc_locations:
                    toc_locations.append(ch1)
                if ch2 not in toc_locations:
                    toc_locations.append(ch2)

        for i in toc_locations:  # Delete lines appearing in toc_locations from chapter_locations
            index = self.chapter_locations.index(i)
            del self.chapter_locations[index]

    def get_end_location(self):
        """
        Find the line number where the book ends if it exists
        :return: int
        """
        # *** END OF THE PROJECT GUTENBERG EBOOK ALICE’S ADVENTURES IN WONDERLAND ***
        ends = ["End of the Project Gutenberg EBook",
                "End of Project Gutenberg's",
                "\*\*\*END OF THE PROJECT GUTENBERG EBOOK",
                "\*\*\* END OF THE PROJECT GUTENBERG EBOOK",
                "\*\*\*END OF THIS PROJECT GUTENBERG EBOOK",
                "\*\*\* END OF THIS PROJECT GUTENBERG EBOOK"]
        pattern = '|'.join(ends)
        regex = re.compile(pattern, re.IGNORECASE)

        for line in self.lines:
            if regex.match(line) is not None:
                return self.lines.index(line)

        # If ending can't be found
        end_location = len(self.lines) - 1
        return end_location

    def get_text_between_chapters(self):
        """
        Extract texts between the declared chapter lines
        :return: list <str>
        """
        chapter_texts = []

        for i, ch1 in enumerate(self.chapter_locations[:-1]):
            ch2 = self.chapter_locations[i + 1]
            chapter_texts.append(self.lines[ch1: ch2])
        return chapter_texts

    @staticmethod
    def zero_pad(number_list):
        """
        Pads numbers with zeroes
        :param number_list: list <int>
        :return: list <str>
        """
        max_num = max(number_list)
        num_digits = len(str(max_num))
        return [str(n).zfill(num_digits) for n in number_list]

    def write_chapters(self):
        """
        Creates a new directory and writes all chapters in different txt files
        :return: None
        """
        chapter_nums = self.zero_pad(range(1, len(self.chapter_texts) + 1))
        filename = os.path.basename(self.filename)
        file = os.path.splitext(filename)[0]
        ext = '.txt'
        output_dir = "mt_chapters/" + file + '-chapters'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for num, chapter in zip(chapter_nums, self.chapter_texts):
            path = output_dir + '/' + num + ext
            chapter = '\n'.join(chapter)
            with open(path, 'w', encoding="utf-8-sig") as f:
                f.write(chapter)
