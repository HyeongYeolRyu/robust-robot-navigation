import numpy as np

class DamageModule:

    def __init__(self,
                 damage_mode='off',
                 sensor_dim=480,
                 sensor_range=4.):

        self.occ = dict()
        self.sensor = dict()

        self.occ['damage_mode'] = damage_mode     # off, gaussian
        self.occ['mask']        = np.zeros(sensor_dim, np.bool)

        self.sensor['dim']   = sensor_dim
        self.sensor['range'] = sensor_range


    def damage_sensor_data(self, sensor_data):

        sensor_data = sensor_data.copy()

        if self.occ['damage_mode'] == 'off':
            sensor_data[self.occ['mask']] = 0.

        elif self.occ['damage_mode'] == 'gaussian':
            pass

        return sensor_data


    def __call__(self, sensor_data):
        sensor_data = self.damage_sensor_data(sensor_data)
        return sensor_data


    def reset(self):
        pass


class ConstDamageModule(DamageModule):

    def __init__(self,
                 damage_mode='off',
                 occ_range=(0, 80),
                 sensor_dim=480,
                 sensor_range=4.):
        
        super(ConstDamageModule, self).__init__(damage_mode, sensor_dim, sensor_range)

        if occ_range[0] < 0 or occ_range[1] >= sensor_dim:
            raise ValueError()

        self.occ['mask'][occ_range[0] : occ_range[1]] = True


class SplitDamageModule(DamageModule):

    def __init__(self,
                 p=0.2,
                 damage_mode='off',
                 splits_num=12,
                 occ_num=1,
                 fix_occ_num=False,
                 sensor_dim=480,
                 sensor_range=4.):
        
        super(SplitDamageModule, self).__init__(damage_mode, sensor_dim, sensor_range)

        if sensor_dim % splits_num != 0:
            raise ValueError()

        self.occ['split_length'] = sensor_dim // splits_num
        self.occ['num'] = occ_num
        self.occ['num_splits'] = splits_num
        self.occ['p'] = p
        self.occ['fix_num'] = fix_occ_num


    def reset(self):
        if np.random.uniform(0, 1) > self.occ['p']:        # Not damaged
            self.occ['mask'][:] = False
            return

        # randomly select the number of occlusion(1 ~ self.occ['num'])
        if self.occ['fix_num'] is False:
            occ_num = np.random.randint(1, self.occ['num'] + 1)
        else:
            occ_num = self.occ['num']

        splits = np.random.choice(self.occ['num_splits'], occ_num, replace=False) * self.occ['split_length']

        for split in splits:
            self.occ['mask'][split: split + self.occ['split_length']] = True


class RandomDamageModule(DamageModule):

    def __init__(self,
                 p=0.2,
                 damage_mode='off',
                 occ_num=1,
                 occ_length=120,
                 fix_occ_num=False,
                 sensor_dim=480,
                 sensor_range=4.):
        super(RandomDamageModule, self).__init__(damage_mode, sensor_dim, sensor_range)

        self.occ['num'] = occ_num
        self.occ['length'] = occ_length
        self.occ['p'] = p
        self.trial = 100
        self.occ['fix_num'] = fix_occ_num

    def reset(self):
        if np.random.uniform(0, 1) > self.occ['p']:        # Not damaged
            self.occ['mask'][:] = False
            return

        if self.occ['fix_num'] is False:
            occ_num = np.random.randint(1, self.occ['num'] + 1)
        else :
            occ_num = self.occ['num']

        total_occ_length = self.occ['length']

        mask_index = []

        for i in range(self.trial):
            if occ_num != 1:
                occ_length = np.random.randint(1, total_occ_length)
                if total_occ_length-occ_length <= occ_num:
                    continue
            else:
                occ_length = total_occ_length

            begin = np.random.randint(0, self.sensor['dim'] - occ_length)
            end = begin + occ_length

            if self.__is_index_overlapped(mask_index, begin, end):
                continue

            mask_index.append((begin, end))

            total_occ_length -= (end - begin)
            occ_num -= 1

            if occ_num == 0:
                assert total_occ_length == 0
                break

        for begin, end in mask_index:
            self.occ['mask'][begin : end] = True


    def __is_index_overlapped(self, mask_index, begin, end):
        for a, b in mask_index:
            if a <= begin < b:
                return True

            if a < end <= b:
                return True

        return False


if __name__ == '__main__':

    sensor_data = np.ones(480, dtype=np.float)

    print('ConstDamageModule ====')
    damage_module = ConstDamageModule(occ_range=[120, 240])
    damage_module.reset()
    print(damage_module(sensor_data))

    print('SplitDamageModule ====')
    damage_module = SplitDamageModule(p=1.0, occ_num=2, fix_occ_num=True)
    damage_module.reset()
    print(damage_module(sensor_data))

    print('RandomDamageModule ====')
    damage_module = RandomDamageModule(p=1.0, occ_num=4, fix_occ_num=True)
    damage_module.reset()
    print(damage_module(sensor_data))
    print(sum(damage_module(sensor_data)))

