def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

feature_dim = 16  #[16, 32,64]

class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("7", "13"), ("12", "16"), ("9", "18"), ("6", "23")]
        # self.scenarios = [("18", "27"), ("20", "5"), ("24", "8"), ("28", "27"), ("30", "20")] # additional
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = feature_dim #16 #32 #64
        self.final_out_channels =feature_dim * 2 #32 # 64 #128
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64       #64
        self.final_out_channels_t =128  #128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        # self.scenarios = [("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")] #additional
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = feature_dim
        self.final_out_channels = feature_dim*2
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t =128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("18", "23"), ("6", "19")]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = feature_dim
        self.final_out_channels = feature_dim *2
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t =128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.sequence_len = 128
        self.scenarios = [("2", "7"), ("0", "6"), ("1", "6"), ("3", "8"), ("4", "5")]
        # self.scenarios = [("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2")] # additional
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels =feature_dim
        self.final_out_channels =feature_dim *2
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t =128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class FD(object):
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        self.scenarios = [ ("0", "3"), ("0", "1"),("2", "1"), ("1", "2"),("2", "3")]
        # self.scenarios = [ ("1", "0"), ("1", "3"), ("3", "0"), ("3", "1"), ("3", "2")] #additional
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        # CNN and RESNET features
        self.mid_channels = feature_dim
        self.final_out_channels = feature_dim * 2
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t = 128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
