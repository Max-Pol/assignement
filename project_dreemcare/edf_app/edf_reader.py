"""
    This file implements EDFReader, a class allowing to read edf+ files.

    After having looked at some EDFReader, I just pick the best I found:
    Vtoto's EDF Reader.

    I added an "anonymization" method.
"""

import re
import datetime
import numpy as np
import sys
import struct
import io


class EDFEndOfData:
    pass


def tal(tal_str):
    """Return a list with (onset, duration, annotation) tuples for an
    EDF+ annotation stream.
    """
    exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
          '(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
          '(\x14(?P<annotation>[^\x00]*))?' + \
          '(?:\x14\x00)'

    def annotation_to_list(annotation):
        return str(annotation, 'utf-8').split('\x14') if annotation else []

    def parse(dic):
        return (
            float(dic['onset']),
            float(dic['duration']) if dic['duration'] else 0.,
            annotation_to_list(dic['annotation']))

    return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]


class EDFReader:

    """
    This classe allows to read an edf file.
    The header of the edfFile is a dict() stored in self.header.
    To read the data, you can use self.records_reader: a generator
    data-record by data record. You can also use self.gen1secData()
    which is a generator 1sec by 1 sec of data.
    """

    def __init__(self, edfFile):
        print(edfFile)
        self.edfFile = io.open(edfFile, 'r+', encoding="latin-1")
        self.header = self.read_header()

    def anonymization(self):

        # replace the EDF subject_id by *
        f = self.edfFile
        f.seek(8)
        f.write("*" * 80)

        # update header
        self.header = self.read_header()

    def read_header(self):
        """
        Read the 256 bytes + number_of_signals * 256 bytes of the header
        """
        h = {}
        f = self.edfFile
        f.seek(0)
        assert f.read(8).split() == ['0']  # version of data format

        # recording info
        h['local_subject_id'] = f.read(80).strip()
        h['local_recording_id'] = f.read(80).strip()

        # parse timestamp
        (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
        (hour, minute, sec) = [int(x) for x in re.findall('(\d+)', f.read(8))]
        h['start_date_time'] = datetime.datetime(year + 2000, month, day,
                                                 hour, minute, sec)

        # misc
        h['header_length'] = int(f.read(8))
        subtype = f.read(44)[:5]
        h['EDF+'] = subtype in ['EDF+C', 'EDF+D']  # yes = edf+ no= edf
        h['contiguous'] = subtype != 'EDF+D'  # continuous ?
        h['nb_records'] = int(f.read(8))  # number of data records
        h['record_duration'] = float(f.read(8))  # duration DataRecord
        nb_signals = h['nb_signals'] = int(f.read(4))  # number of signals

        # read signals info
        signals = list(range(nb_signals))
        h['label'] = [f.read(16).strip() for n in signals]
        h['transducer_type'] = [f.read(80).strip() for n in signals]
        h['physical_dimension'] = [f.read(8).strip() for n in signals]
        h['physical_min'] = np.asarray([float(f.read(8)) for n in signals])
        h['physical_max'] = np.asarray([float(f.read(8)) for n in signals])
        h['digital_min'] = np.asarray([float(f.read(8)) for n in signals])
        h['digital_max'] = np.asarray([float(f.read(8)) for n in signals])
        h['prefiltering'] = [f.read(80).strip() for n in signals]
        h['nb_samples_per_record'] = [int(f.read(8)) for n in signals]
        try:
            h['frequency'] = [h['nb_samples_per_record'][n] /
                              h['record_duration'] for n in signals]
        except:
            h['frequency'] = [h['nb_samples_per_record'][n] for n in signals]
        f.read(32 * nb_signals)  # reserved

        assert f.tell() == h['header_length']  # Check local position

        # calculate ranges for rescaling
        phys_range = h['physical_max'] - h['physical_min']
        dig_range = h['digital_max'] - h['digital_min']
        # assert np.all(abs(phys_range) > 0)  # To check
        # assert np.all(abs(dig_range) > 0)  # To check
        h['gain'] = phys_range / dig_range
        return h

    def print_header_summary(self):
        """
        Return a summary of mains information of the header
        """
        Summary = "Subject ID  : " +\
            self.header['local_subject_id'] + "\nRecord ID   : " +\
            self.header['local_recording_id'] + "\nDate        : " +\
            str(self.header['start_date_time']) + "\nEDF+        : " +\
            str(self.header['EDF+']) + "\nContinu     : " +\
            str(self.header['contiguous']) + "\nNb Records  : " +\
            str(self.header['nb_records']) + "\nDuration rec: " +\
            str(self.header['record_duration']) + "\nNb signals  : " +\
            str(self.header['nb_signals']) + "\nSignals     : " +\
            str(self.header['label']) + "\nFrequencies : " +\
            str(self.header['frequency']) + "\nsample/rec  : " +\
            str(self.header['nb_samples_per_record'])
        print(Summary)
        return Summary

    def print_info_signal(self, i):
        """Return main informations relative to signal i"""
        info = {}
        for key, value in list(self.header.items()):
            if type(value) == list:
                info[key] = value[i]
        return info

    def read_next_raw_record(self):
        """
        Read bytes corresponding to the next data record and
        return a list containing raw bytes for each signal
        """
        raw_bytes = []
        for nbsamp in self.header['nb_samples_per_record']:
            samples = self.edfFile.read(nbsamp * 2)
            if len(samples) != nbsamp * 2:
                raise EDFEndOfData
            raw_bytes.append(samples)
        return raw_bytes

    def convert_record(self, raw_record):
        """Convert a raw record to a (time, signals, events) tuple based on
        information in the header.
        time is the beginning of the data record
        signals are the signals values during the data records
        events are the eventual events happening during this data_record
        """
        h = self.header
        time = float('nan')
        signals = []
        events = []
        for (i, samples) in enumerate(raw_record):  # browsing each signal
            if h['label'][i] == 'EDF Annotations':  # Annotations
                ann = tal(samples)
                time = ann[0][0]  # Data record time
                events.extend(ann[1:])
            else:  # ordinary signal 2bytes per sample
                # 2-byte little-endian integers
                dig = np.fromstring(samples, '<i2').astype(np.float32)
                phys = (dig - h['digital_min'][i]) * h['gain'][i] +\
                    h['physical_min'][i]
                signals.append(phys)
        return time, signals, events

    def records_reader(self):
        """
        Data record by data Record generator, return one data record
        as a tuple (time, signals, events) (cf. convert_record)
        """
        self.edfFile.seek(self.header['header_length'])
        try:
            while True:
                yield self.convert_record(self.read_next_raw_record())
        except EDFEndOfData:
            pass

    def gen_1_sec_data(self):
        """
        Return 1 sec data for every signal as a list of signals
        """
        assert self.header['record_duration'] - \
            int(self.header['record_duration']) == 0
        record_duration = int(self.header['record_duration'])
        sample_per_rec = self.header['nb_samples_per_record']
        nb_signals = self.header['nb_signals']
        # Checking if nb_sample_per_record/record_duration is an int
        assert [x % record_duration == 0 for x in sample_per_rec] == \
            [True] * nb_signals
        new_sample_per_rec = [x / record_duration for x in sample_per_rec]
        self.edfFile.seek(self.header['header_length'])
        try:
            while True:
                data = self.convert_record(self.read_next_raw_record())[1]
                for i in range(record_duration):
                    yield [x[i * nc: (i + 1) * nc] for x, nc in
                           zip(data, new_sample_per_rec)]
        except EDFEndOfData:
            pass

    def get_data(self):
        """
        Return all the signals as a matrix and set self.data as this matrix
        """
        if hasattr(self, 'data'):
            return self.data
        else:
            gen = self.records_reader()
            h = self.header
            nb_signals = self.get_nb_signal()
            len_data = [h['nb_samples_per_record'][i] *
                        h['nb_records'] for i in range(nb_signals)]
            data = [[0 for i in range(len_data[j])] for j in range(nb_signals)]
            for i in range(h['nb_records']):
                tmp = gen.next()[1]
                for j in range(nb_signals):
                    data[j][i * h['nb_samples_per_record'][j]:(i + 1) *
                            h['nb_samples_per_record'][j]] = tmp[j]
            self.data = data
        return data

    def get_signal(self, i):
        """
        Return signal i as a vector
        """
        if not hasattr(self, 'data'):
            return self.get_signal_i_fast(i)
        else:
            return self.data[i]

    def get_signal_i_fast(self, i):
        len_db = sum(self.header['nb_samples_per_record'])
        offset_i = sum(self.header['nb_samples_per_record'][:i])
        nb_sample_per_record_i = self.header['nb_samples_per_record'][i]
        data_i = [0 for j in range(nb_sample_per_record_i)]
        dig_min_i = self.header['digital_min'][i]
        gain_i = self.header['gain'][i]
        phys_min_i = self.header['physical_min'][i]
        for i in range(self.header['nb_records']):  # FIX: "-2"
            self.edfFile.seek(self.header['header_length'] +
                              i * len_db * 2 + offset_i * 2)
            raw_data = self.edfFile.read(nb_sample_per_record_i * 2)
            # print(len(bytes(raw_data, encoding='latin-1')))
            # print(nb_sample_per_record_i)
            if (len(bytes(raw_data, encoding='latin-1')) == 256):
                dig = struct.unpack('h' * nb_sample_per_record_i,
                                    bytes(raw_data, encoding='latin-1'))
                phys = (dig - dig_min_i) * gain_i + phys_min_i
                data_i[i * nb_sample_per_record_i: (i + 1) *
                       nb_sample_per_record_i] = phys
        return data_i

    def get_annotations(self):
        """
        If EDF+ get all annotations (except those relative
        to data record start time)
        """
        annotations = []
        records_reader = self.records_reader()
        for m in records_reader:
            for i in range(len(m[2])):
                annotations.append(m[2][i])
        # we delete the last annotations that corresponds to wake
        self.annotations = annotations[:-5]

    def get_nb_signal(self):
        """
        return the number of signal in the edf
        """
        return self.header['nb_signals'] - \
            self.header['label'].count('EDF Annotations')


if __name__ == "__main__":
    red = EDFReader(sys.argv[1])
    red.print_header_summary()
    gen = red.records_reader()
    data = red.get_signal(1)
