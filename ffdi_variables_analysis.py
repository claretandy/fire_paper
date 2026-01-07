import os
import sys
os.environ['USE_PYGEOS'] = '0'

import dask.array.ma
import iris
import glob
import pandas as pd
import numpy as np
import numpy.ma as ma
import cftime
from iris.util import unify_time_units
import geopandas as gpd
sys.path.append('~/PycharmProjects/impactstoolbox')
from impactstoolbox.tools import helper_functions as hf
import subprocess
import warnings
warnings.simplefilter('ignore')
import pdb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time

scratchdir = "/data/scratch/andrew.hartley/"
datadir = "/data/users/andrew.hartley/"


def get_ens_df():
    #ensemble members for rcp2.6 historical runs
    hist_rcp2_6=['akowa', 'akowb', 'akowc', 'akowe', 'akowf', 'akowh', 'akowi', 'akowj', 'akowk', 'akowl',
                'akowm', 'akown', 'akowo', 'akowp', 'akowq', 'akoxa', 'akoxb', 'akoxc', 'akoxd', 'akoxe',
                'akoxg', 'akoxi', 'akoxj', 'akoxl', 'akoxm', 'akoxn', 'akoxo', 'akoxp', 'akoxq', 'akoya',
                'akoyb', 'akoyc', 'akoye', 'akoyf', 'akoyg', 'akoyh', 'akoyi', 'akoyj', 'akoyk', 'akoyl',
                'akoym', 'akoyn', 'akoyq', 'akoza', 'akozb', 'akozc', 'akozd', 'akoze', 'akozg', 'akozh',
                'akozi', 'akozj', 'akozl', 'akozn', 'akozo', 'akozp', 'akozq']
    #ensemble members for rcp2.6 future runs
    fut_rcp2_6=['akuka', 'akukb', 'akukc', 'akuke', 'akukf', 'akukh','akuki', 'akukj', 'akukk', 'akukl',
                'akukm', 'akukn', 'akuko', 'akukp', 'akukq', 'akula', 'akulb', 'akulc', 'akuld', 'akule',
                'akulg', 'akuli', 'akulj', 'akull', 'akulm', 'akuln', 'akulo', 'akulp', 'akulq', 'akuma',
                'akumb', 'akumc', 'akume', 'akumf', 'akumg', 'akumh', 'akumi', 'akumj', 'akumk', 'akuml',
                'akumm', 'akumn', 'akumq', 'akuna', 'akunb', 'akunc', 'akund', 'akune', 'akung', 'akunh',
                'akuni', 'akunj', 'akunl', 'akunn', 'akuno', 'akunp', 'akunq']
    #ensemble members for rcp8.5 historical runs (same as for rcp2.6)
    hist_rcp8_5=['akowa', 'akowb', 'akowc', 'akowe', 'akowf', 'akowh', 'akowi', 'akowj', 'akowk', 'akowl',
                 'akowm', 'akown', 'akowo', 'akowp', 'akowq', 'akoxa', 'akoxb', 'akoxc', 'akoxd', 'akoxe',
                 'akoxg', 'akoxi', 'akoxj', 'akoxl', 'akoxm', 'akoxn', 'akoxo', 'akoxp', 'akoxq', 'akoya',
                 'akoyb', 'akoyc', 'akoye', 'akoyf', 'akoyg', 'akoyh', 'akoyi', 'akoyj', 'akoyk', 'akoyl',
                 'akoym', 'akoyn', 'akoyq', 'akoza', 'akozb', 'akozc', 'akozd', 'akoze', 'akozg', 'akozh',
                 'akozi', 'akozj', 'akozl', 'akozn', 'akozo', 'akozp', 'akozq']
    #ensemble members for rcp8.5 future runs
    fut_rcp8_5=['aldpa', 'aldpb', 'aldpc', 'aldpe', 'aldpf', 'aldph', 'aldpi', 'aldpj', 'aldpk', 'aldpl',
                'aldpm', 'aldpn', 'aldpo', 'aldpp', 'aldpq', 'aldqa', 'aldqb', 'aldqc', 'aldqd', 'aldqe',
                'aldqg', 'aldqi', 'aldqj', 'aldql', 'aldqm', 'aldqn', 'aldqo', 'aldqp', 'aldqq', 'aldra',
                'aldrb', 'aldrc', 'aldre', 'aldrf', 'aldrg', 'aldrh', 'aldri', 'aldrj', 'aldrk', 'aldrl',
                'aldrm', 'aldrn', 'aldrq', 'aldsa', 'aldsb', 'aldsc', 'aldsd', 'aldse', 'aldsg', 'aldsh',
                'aldsi', 'aldsj', 'aldsl', 'aldsn', 'aldso', 'aldsp', 'aldsq']

    # Create a pandas dataframe from the above lists
    df2p6 = pd.DataFrame({'rcp': 'rcp2.6', 'hist': hist_rcp2_6, 'fut': fut_rcp2_6})
    df8p5 = pd.DataFrame({'rcp': 'rcp8.5', 'hist': hist_rcp8_5, 'fut': fut_rcp8_5})
    df = pd.concat([df2p6, df8p5], ignore_index=True)
    df['gwl_1_5_start'] = np.nan
    df['gwl_2_start'] = np.nan
    df['gwl_4_start'] = np.nan
    df = df.set_index('fut')

    # Now read the GWL data
    gwl_dir = f'{datadir}Projects/FFDI_Paper/GlobalWarmingLevels/*/*.txt'
    gwl_files = glob.glob(gwl_dir)
    for f in gwl_files:
        rcp = os.path.dirname(f).split('/')[-1].replace('_', '.')
        gwl = 'gwl_' + os.path.basename(f).split('gwl_')[1].split('.txt')[0].replace('_deg', '')
        print(rcp, gwl)
        # Read the csv file
        gwldf = pd.read_csv(f, header=None, names=['fut', gwl+'_start', 'end_yr'], na_values=['nan', ' nan', ' nan '], dtype={'fut': 'string', gwl+'_start':'Int64', 'end_yr':'Int64'})
        gwldf = gwldf.set_index('fut')
        # Update df with the contents of gwldf
        df.update(gwldf)

    # Removes the index 'fut', so it is part of the table again
    df.reset_index(inplace=True)
    # Reorder the columns to something more sensible
    df = df.reindex(columns=['rcp', 'hist', 'fut', 'gwl_1_5_start', 'gwl_2_start', 'gwl_4_start'])

    return df


def check_ens_numbers():
    'do stuff'
    indir = '/scratch/hadin/fire/ffdi_inputs/'
    files = sorted(glob.glob(indir+'*/*'))
    ens = sorted(list(set([os.path.basename(f).split('_')[5] for f in files])))
    dbl = 0
    rcp2p6 = 0
    rcp8p5 = 0
    for em in ens:
        emfiles = glob.glob(indir+'*/*'+em+'*')
        filercps = set([os.path.dirname(emf).split('/')[-1] for emf in emfiles])
        print(em, ':', filercps)
        dbl = dbl + 1 if len(filercps) > 1 else dbl
        rcp2p6 = rcp2p6 + 1 if filercps == {'rcp2_6'} else rcp2p6
        rcp8p5 = rcp8p5 + 1 if filercps == {'rcp8_5'} else rcp8p5

    print('dbl:', dbl, '; rcp2.6:', rcp2p6, '; rcp8.5:', rcp8p5)


def periodConstraint(cube, t1, t2, greedy=False):
    # Constrains the cube according to min and max datetimes
    def make_time_func(t1m, t2m, greedy=False):
        def tfunc(cell):
            if greedy:
                return t1m <= cell.point <= t2m
            else:
                return t1m <= cell.point < t2m
        return tfunc

    tfunc = make_time_func(t1, t2, greedy=greedy)
    tconst = iris.Constraint(time=tfunc)
    ocube = cube.extract(tconst)

    return ocube


def load_files(file_list, start_yr):

    if np.isnan(start_yr):
        return None
    else:
        varnames = ['relative_humidity', 'air_temperature', 'precipitation_in_mpersec', 'windspeed_in_km_per_hr', 'soil_moisture_content_top_metre'] # 'vol_SMC_at_field_capacity'
        start_dt = cftime.Datetime360Day(start_yr, 1, 1)
        end_dt = cftime.Datetime360Day(start_yr + 20, 1, 1)
        cubelist = iris.cube.CubeList([])
        for var in varnames:
            print('   ... ' + var, end=' ... ')
            cubes = iris.load(file_list, iris.Constraint(name=var))
            unify_time_units(cubes)
            varcubes = iris.cube.CubeList([])
            for cube in cubes:
                cube.remove_coord('forecast_reference_time')
                varcubes.append(cube)

            varcube = varcubes.concatenate_cube()
            varcube = periodConstraint(varcube, start_dt, end_dt)

            cubelist.append(varcube)
            print('done')

        return cubelist


def get_variables_by_ffdi_theshold(df, mint, maxt, ofile, country_list=['Australia', 'United States of America', 'Brazil']):
    """
    Gets a massive dataframe of FFDI value (>=12) with each variable per grid cell per day
    :param df: Dataframe that tells us for each ens member what the start year is for each GWL
    :param mint: masks < mint
    :param maxt: if no maximum, use np.inf
    :return:
    """

    # Root dir of the data
    indir = f'{scratchdir}fire/ffdi_input_variables/'

    # Baseline start year
    bl_start_yr = 1986

    vars = ['relative_humidity', 'soil_moisture_content_top_metre', 'precipitation_in_mpersec',
            'windspeed_in_km_per_hr', 'air_temperature']

    # Make empty output dataframe
    column_names = ['t', 'x', 'y', 'gwl', 'ensemble_member', 'rcp', 'FFDI', 'air_temperature', 'precipitation_in_mpersec', 'relative_humidity', 'soil_moisture_content_top_metre', 'windspeed_in_km_per_hr']
    df_final = pd.DataFrame(columns=column_names)

    # Loop through each row of the table (contains the RCP & future jobid)
    for i, row in df.iterrows():
        print(i, row['rcp'], row['hist'], row['fut'])
        rcp = row['rcp'].replace('.', '_')

        # ofile = f'{datadir}Projects/FFDI_Paper/data/odf_bigdata_country_{mint}_to_{maxt}_temp{i}.csv'
        ofile_temp = ofile.replace('.csv', f'_temp{i}.csv')
        if os.path.isfile(ofile_temp):
            df_final = pd.read_csv(ofile_temp)
            continue
        else:
            last_ofile = ofile.replace('.csv', f'_temp{i-1}.csv')
            # last_ofile = f'/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_bigdata_country_{mint}_to_{maxt}_temp{i-1}.csv'
            if os.path.isfile(last_ofile):
                df_final = pd.read_csv(last_ofile)

            ##########
            # Get the pre-calculated FFDI (saved as a timeseries of baseline + future, based on future ens_member_id
            print('   Loading FFDI ... ')
            ffdi_filelist = glob.glob('/data/scratch/andrew.hartley/fire/ffdi_output/'+rcp+'/ffdi_'+rcp+'_'+row['hist']+'*.nc')
            ffdi_filelist.extend(glob.glob('/data/scratch/andrew.hartley/fire/ffdi_output/'+rcp+'/ffdi_'+rcp+'_'+row['fut']+'*.nc'))

            # Load the files into an iris cubelist
            ffdi_cubes = iris.load(ffdi_filelist)  # includes baseline and future FFDI

            # Remove some things to allow for the cube to be concatenated
            for j in range(len(ffdi_cubes)):
                ffdi_cubes[j].remove_coord('forecast_period')
                ffdi_cubes[j].remove_coord('year')

            # Now unify time coords and concatenate
            unify_time_units(ffdi_cubes)
            ffdi_cube = ffdi_cubes.concatenate_cube()

            # Get the burnable mask
            burnable = hf.get_burnable_mask(ffdi_cube)

            # Now get FFDI variables
            var_filelist = sorted(glob.glob(f"{indir}{rcp}/ffdi_input_variables_{rcp}_{row['hist']}_*"))
            var_filelist.extend(sorted(glob.glob(f"{indir}{rcp}/ffdi_input_variables_{rcp}_{row['fut']}_*")))

            # Make a dictionary of cubes for all variables  ...
            all_var_cubes = {}
            for v in vars:
                print(f"   ... {v}")
                print(var_filelist)
                # Do all the same loading things ...
                var_cubes = iris.load(var_filelist, v)  # baseline and future variable
                for j in range(len(var_cubes)):
                    for coord in ['forecast_reference_time', 'forecast_period', 'year']:
                        try:
                            var_cubes[j].remove_coord(coord)
                        except:
                            continue

                unify_time_units(var_cubes)
                all_var_cubes[v] = var_cubes.concatenate_cube()

            # Time subset of ffdi and variables
            tcon_dict = {}
            # Baseline
            hist_start = cftime.Datetime360Day(bl_start_yr, 1, 1)
            hist_end = cftime.Datetime360Day(bl_start_yr + 20, 1, 1)
            hist_tcon = iris.Constraint(time=lambda cell: hist_start <= cell.point < hist_end)
            tcon_dict['Baseline'] = hist_tcon

            # GWL 1.5C
            gwl1p5_start = cftime.Datetime360Day(row['gwl_1_5_start'], 1, 1)
            gwl1p5_end = cftime.Datetime360Day(row['gwl_1_5_start'] + 20, 1, 1)
            gwl1p5_tcon = iris.Constraint(time=lambda cell: gwl1p5_start <= cell.point < gwl1p5_end)
            tcon_dict['GWL1.5'] = gwl1p5_tcon

            # GWL 2C
            try:
                gwl2_start = cftime.Datetime360Day(row['gwl_2_start'], 1, 1)
                gwl2_end = cftime.Datetime360Day(row['gwl_2_start'] + 20, 1, 1)
                gwl2_tcon = iris.Constraint(time=lambda cell: gwl2_start <= cell.point < gwl2_end)
                tcon_dict['GWL2'] = gwl2_tcon
            except:
                print(f"{row['rcp']} & {row['fut']} doesn\'t reach 2C GWL")
                pass

            # GWL 4C
            try:
                gwl4_start = cftime.Datetime360Day(row['gwl_4_start'], 1, 1)
                gwl4_end = cftime.Datetime360Day(row['gwl_4_start'] + 20, 1, 1)
                gwl4_tcon = iris.Constraint(time=lambda cell: gwl4_start <= cell.point < gwl4_end)
                tcon_dict['GWL4'] = gwl4_tcon
            except:
                print(f"{row['rcp']} & {row['fut']} doesn\'t reach 4C GWL")
                pass

            # Loop through the time constraints and extract all the grid cells with data
            print(f'   Extracting time slices')
            dfout = {}
            for key, tcon in tcon_dict.items():
                print(f'   ... {key}: FFDI, ')
                # Extract the days in the time constraint
                ffdi_tconned = ffdi_cube.extract(tcon)
                # Mask according to the min and max thresholds
                ffdi_tconned.data = ma.masked_where((ffdi_tconned.data < mint) | (ffdi_tconned.data >= maxt), ffdi_tconned.data)
                # Combined burnable mask with masked data
                burnable_allts = np.broadcast_to(ma.getmask(burnable.data), ffdi_tconned.shape)
                fullmask = burnable_allts | ma.getmask(ffdi_tconned.data)
                # Do the masking and convert to pandas dataframe
                tmp = mask2df(ffdi_tconned, fullmask, 'FFDI', key, row['fut'], row['rcp'], country_list=country_list)

                if isinstance(dfout, dict):
                    dfout = tmp.copy()
                else:
                    dfout = pd.concat([dfout, tmp])

                for k in all_var_cubes.keys():
                    print(f"   ... {key}: {k}")
                    var_tconned = all_var_cubes[k].extract(tcon)
                    varout = mask2df(var_tconned, fullmask, k, key, row['fut'], row['rcp'], country_list=country_list)
                    # varout = mask2df(var_tconned, ~ma.getmask(ffdi_tconned.data), k, key, row['fut'], row['rcp'])
                    dfout = pd.concat([dfout, varout])

            # NB: dfout contains all the GWLs (i.e. time slices) for this ens member
            # This pivot widens the var column, to give a column per category
            # Therefore each row is a unique combination of t-x-y-gwl
            odf_wide = pd.pivot_table(dfout, values="value", index=['t', 'x', 'y', 'gwl', 'names', 'ensemble_member', 'rcp'], columns=["var"])
            odf_wide.reset_index(inplace=True)

            pdb.set_trace()
            ### Probably need to sample here, in order to reduce the size of the output ###
            # odf_wide_ss = odf_wide.sample(n=10000, replace=False, axis='index', ignore_index=True)
            odf_wide_ss = odf_wide.groupby(['gwl', 'names']).sample(frac=0.001, replace=False)  # axis='index', ignore_index=True
            # odf_wide_ss = odf_wide.groupby(['gwl', 'names']).sample(n=100, replace=True)
            # Concatenate with existing output, and on to the next ensemble member
            df_final = pd.concat([df_final, odf_wide_ss])
            # Now save the current state of df_final to a temporary file
            df_final.to_csv(ofile_temp)

    df_final.to_csv(ofile)

    return df_final


def get_variables_by_ffdi_theshold_nosample(df, mint, maxt, ofile, country_list=['Australia', 'United States of America', 'Brazil']):
    """
    Gets a massive dataframe of FFDI value (>=12) with each variable per grid cell per day, WITHOUT SAMPLING
    :param df: Dataframe that tells us for each ens member what the start year is for each GWL
    :param mint: masks < mint
    :param maxt: if no maximum, use np.inf
    :return:
    """

    # Root dir of the data
    indir = f'{scratchdir}fire/ffdi_input_variables/'

    # Baseline start year
    bl_start_yr = 1986

    vars = ['relative_humidity', 'soil_moisture_content_top_metre', 'precipitation_in_mpersec',
            'windspeed_in_km_per_hr', 'air_temperature']

    # Make empty output dataframe
    column_names = ['t', 'x', 'y', 'gwl', 'ensemble_member', 'rcp', 'FFDI', 'air_temperature', 'precipitation_in_mpersec', 'relative_humidity', 'soil_moisture_content_top_metre', 'windspeed_in_km_per_hr']
    df_final = pd.DataFrame(columns=column_names)

    # Loop through each row of the table (contains the RCP & future jobid)
    for i, row in df.iterrows():
        print(i, row['rcp'], row['hist'], row['fut'])
        rcp = row['rcp'].replace('.', '_')

        # ofile = f'{datadir}Projects/FFDI_Paper/data/odf_bigdata_country_{mint}_to_{maxt}_temp{i}.csv'
        ofile_temp = ofile.replace('.csv', f'_temp{i}.csv')
        ofile_ens = ofile.replace('.csv', f'_ens-{i}.csv')

        if os.path.isfile(ofile_temp):
            df_final = pd.read_csv(ofile_temp)
            continue
        else:
            last_ofile = ofile.replace('.csv', f'_temp{i-1}.csv')
            # last_ofile = f'/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_bigdata_country_{mint}_to_{maxt}_temp{i-1}.csv'
            if os.path.isfile(last_ofile):
                df_final = pd.read_csv(last_ofile)

            ##########
            # Get the pre-calculated FFDI (saved as a timeseries of baseline + future, based on future ens_member_id
            print('   Loading FFDI ... ')
            ffdi_filelist = glob.glob('/data/scratch/andrew.hartley/fire/ffdi_output/'+rcp+'/ffdi_'+rcp+'_'+row['hist']+'*.nc')
            ffdi_filelist.extend(glob.glob('/data/scratch/andrew.hartley/fire/ffdi_output/'+rcp+'/ffdi_'+rcp+'_'+row['fut']+'*.nc'))

            # Load the files into an iris cubelist
            ffdi_cubes = iris.load(ffdi_filelist)  # includes baseline and future FFDI

            # Remove some things to allow for the cube to be concatenated
            for j in range(len(ffdi_cubes)):
                ffdi_cubes[j].remove_coord('forecast_period')
                ffdi_cubes[j].remove_coord('year')

            # Now unify time coords and concatenate
            unify_time_units(ffdi_cubes)
            ffdi_cube = ffdi_cubes.concatenate_cube()

            # Get the burnable mask
            burnable = hf.get_burnable_mask(ffdi_cube)

            # Now get FFDI variables
            var_filelist = sorted(glob.glob(f"{indir}{rcp}/ffdi_input_variables_{rcp}_{row['hist']}_*"))
            var_filelist.extend(sorted(glob.glob(f"{indir}{rcp}/ffdi_input_variables_{rcp}_{row['fut']}_*")))

            # Make a dictionary of cubes for all variables  ...
            all_var_cubes = {}
            for v in vars:
                print(f"   ... {v}")
                print(var_filelist)
                # Do all the same loading things ...
                var_cubes = iris.load(var_filelist, v)  # baseline and future variable
                for j in range(len(var_cubes)):
                    for coord in ['forecast_reference_time', 'forecast_period', 'year']:
                        try:
                            var_cubes[j].remove_coord(coord)
                        except:
                            continue

                unify_time_units(var_cubes)
                all_var_cubes[v] = var_cubes.concatenate_cube()

            # Time subset of ffdi and variables
            tcon_dict = {}
            # Baseline
            hist_start = cftime.Datetime360Day(bl_start_yr, 1, 1)
            hist_end = cftime.Datetime360Day(bl_start_yr + 20, 1, 1)
            hist_tcon = iris.Constraint(time=lambda cell: hist_start <= cell.point < hist_end)
            tcon_dict['Baseline'] = hist_tcon

            # GWL 1.5C
            gwl1p5_start = cftime.Datetime360Day(row['gwl_1_5_start'], 1, 1)
            gwl1p5_end = cftime.Datetime360Day(row['gwl_1_5_start'] + 20, 1, 1)
            gwl1p5_tcon = iris.Constraint(time=lambda cell: gwl1p5_start <= cell.point < gwl1p5_end)
            tcon_dict['GWL1.5'] = gwl1p5_tcon

            # GWL 2C
            try:
                gwl2_start = cftime.Datetime360Day(row['gwl_2_start'], 1, 1)
                gwl2_end = cftime.Datetime360Day(row['gwl_2_start'] + 20, 1, 1)
                gwl2_tcon = iris.Constraint(time=lambda cell: gwl2_start <= cell.point < gwl2_end)
                tcon_dict['GWL2'] = gwl2_tcon
            except:
                print(f"{row['rcp']} & {row['fut']} doesn\'t reach 2C GWL")
                pass

            # GWL 4C
            try:
                gwl4_start = cftime.Datetime360Day(row['gwl_4_start'], 1, 1)
                gwl4_end = cftime.Datetime360Day(row['gwl_4_start'] + 20, 1, 1)
                gwl4_tcon = iris.Constraint(time=lambda cell: gwl4_start <= cell.point < gwl4_end)
                tcon_dict['GWL4'] = gwl4_tcon
            except:
                print(f"{row['rcp']} & {row['fut']} doesn\'t reach 4C GWL")
                pass

            # Loop through the time constraints and extract all the grid cells with data
            print(f'   Extracting time slices')
            dfout = {}
            for key, tcon in tcon_dict.items():
                print(f'   ... {key}: FFDI, ')
                # Extract the days in the time constraint
                ffdi_tconned = ffdi_cube.extract(tcon)
                # Mask according to the min and max thresholds
                ffdi_tconned.data = ma.masked_where((ffdi_tconned.data < mint) | (ffdi_tconned.data >= maxt), ffdi_tconned.data)
                # Combined burnable mask with masked data
                burnable_allts = np.broadcast_to(ma.getmask(burnable.data), ffdi_tconned.shape)
                fullmask = burnable_allts | ma.getmask(ffdi_tconned.data)
                # Do the masking and convert to pandas dataframe
                tmp = mask2df(ffdi_tconned, fullmask, 'FFDI', key, row['fut'], row['rcp'], country_list=country_list)

                if isinstance(dfout, dict):
                    dfout = tmp.copy()
                else:
                    dfout = pd.concat([dfout, tmp])

                for k in all_var_cubes.keys():
                    print(f"   ... {key}: {k}")
                    var_tconned = all_var_cubes[k].extract(tcon)
                    varout = mask2df(var_tconned, fullmask, k, key, row['fut'], row['rcp'], country_list=country_list)
                    # varout = mask2df(var_tconned, ~ma.getmask(ffdi_tconned.data), k, key, row['fut'], row['rcp'])
                    dfout = pd.concat([dfout, varout])

                # Only do the baseline for testing
                # break

                # Therefore each row is a unique combination of t-x-y-gwl
                odf_wide = pd.pivot_table(dfout, values="value", index=['t', 'x', 'y', 'gwl', 'names', 'ensemble_member', 'rcp'], columns=["var"])
                odf_wide.reset_index(inplace=True)
                ofile_ens_period = ofile_ens.replace('.csv', f'_allpts-{key}.csv')
                print('   ... saving ensemble member data to ', ofile_ens_period)
                odf_wide.to_csv(ofile_ens_period)
    #
    #         ### Probably need to sample here, in order to reduce the size of the output ###
    #         # odf_wide_ss = odf_wide.sample(n=10000, replace=False, axis='index', ignore_index=True)
    #         odf_wide_ss = odf_wide.groupby(['gwl', 'names']).sample(frac=0.001, replace=False)  # axis='index', ignore_index=True
    #         # odf_wide_ss = odf_wide.groupby(['gwl', 'names']).sample(n=100, replace=True)
    #         # Concatenate with existing output, and on to the next ensemble member
    #         df_final = pd.concat([df_final, odf_wide_ss])
    #         # Now save the current state of df_final to a temporary file
    #         df_final.to_csv(ofile_temp)
    #
    # df_final.to_csv(ofile)

    return


def save_to_feather(df, filename="data.feather"):
    df.to_feather(filename)


def load_from_feather(filename="data.feather"):
    return pd.read_feather(filename)


def sample_data(frac=0.0001, nsamp=300, set=1, iteration=1):
    '''
    Loads in the big data csv, and samples it down to a more manageable size
    Allows for many iterations of sampling to get a sense of variability
    :param frac:
    :param set:
    :param iteration:
    :return:
    '''
    datadir = '/data/users/andrew.hartley/Projects/FFDI_Paper/data/'
    infiles = sorted(glob.glob(f'{datadir}odf_bigdata_3countries_12_to_inf_ens-*_allpts-Baseline.csv'))

    # Sample from each Baseline file, select the same locations in the 4C file and then concatenate into one final dataframe
    df_final = pd.DataFrame()
    for f in infiles:

        # If the same ensemble member also has a GWL4 file, then select the same grid cells from the Baseline file
        f4c = f.replace('Baseline', 'GWL4')
        if os.path.isfile(f4c):

            print('Loading ', f)
            feather_file = f.replace('.csv', '.feather')
            if os.path.isfile(feather_file):
                print('   Loading from feather file ', feather_file)
                df = load_from_feather(feather_file)
            else:
                df = pd.read_csv(f)
                print('   Saving to feather file ', feather_file)
                save_to_feather(df, feather_file)

            # df_samp = df.groupby(['x', 'y', 'names']).sample(frac=frac, replace=False)
            # df_samp = df.groupby(['names']).sample(frac=frac, replace=False)
            df_samp = df.groupby(['names']).sample(n=nsamp, replace=False)
            df_final = pd.concat([df_final, df_samp])

            print('Also loading ', f4c)
            feather_file_f4c = f4c.replace('.csv', '.feather')
            if os.path.isfile(feather_file_f4c):
                print('   Loading from feather file ', feather_file_f4c)
                df_4c = load_from_feather(feather_file_f4c)
            else:
                df_4c = pd.read_csv(f4c)
                print('   Saving to feather file ', feather_file_f4c)
                save_to_feather(df_4c, feather_file_f4c)

            # Remove cells that are not GWL4
            df_4c = df_4c[df_4c['gwl'] == 'GWL4']

            # Following code uses the same grid cells as sampled in the Baseline file (decided against using this approach)
            # gridcells = df_samp[['x', 'y', 'names']].drop_duplicates()
            # # Merge to get only the same grid cells in the GWL4 dataframe
            # df_4c_samp = pd.merge(df_4c, gridcells, on=['x', 'y', 'names'], how='inner')
            # # Then sample to the same fraction
            # df_4c_xybase_samp = df_4c_samp.groupby(['x', 'y', 'names']).sample(frac=frac, replace=False)

            # df_4c_samp = df_4c.groupby(['names']).sample(frac=frac, replace=False)
            df_4c_samp = df_4c.groupby(['names']).sample(n=nsamp, replace=False)
            df_final = pd.concat([df_final, df_4c_samp])

    # Do some classification ...
    df_final['ffdi_class'] = 0
    class12_24 = (df_final.FFDI >= 12) & (df_final.FFDI < 24)
    df_final['ffdi_class'].loc[class12_24] = '12 <= FFDI < 24'
    class_gt24 = (df_final.FFDI >= 24)
    df_final['ffdi_class'].loc[class_gt24] = 'FFDI >= 24'

    # Save the output
    outfile = f'{datadir}Sampled/set{set}/sampled_bigdata_3countries_12_to_inf_sampled-n{nsamp}_set-{set}_iteration-{iteration}.csv'
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    df_final.to_csv(outfile)
    print('Saved ', outfile)
    return  df_final


def get_variable_importance(df_sample):

    # Only when FFDI >= 24

    output = []
    countries = ['Australia', 'Brazil', 'United States of America']  # 'South Africa',

    # for ffdi_class in ['12 <= FFDI < 24', 'FFDI >= 24']:
    for gwl in ['Baseline', "GWL4"]:
        for country in countries:
            print(gwl, country)
            # Subset the big dataframe
            # mysample = df_sample[(df_sample.names.isin([country])) & (df_sample.gwl.isin([gwl]))].sample(frac=0.1, replace=False)
            mysample = df_sample[(df_sample.names.isin([country])) & (df_sample.gwl.isin([gwl]))]
            # mysample['ffdi_class_int'] = 0
            # mysample['ffdi_class_int'].loc[mysample['ffdi_class'] == 'FFDI >= 24'] = 1

            # Extract training and test data
            feature_names = ['air_temperature', 'precipitation_in_mpersec', 'relative_humidity',
                             'soil_moisture_content_top_metre', 'windspeed_in_km_per_hr']
            # pdb.set_trace()
            X_features = mysample[feature_names]
            y_classes = mysample[['FFDI']].astype(int)
            # X_train, X_test, y_train, y_test = train_test_split(X_features, y_classes, stratify=mysample['ffdi_class_int'], random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_features, y_classes, random_state=42)

            # Run the RF classifier
            forest = RandomForestClassifier(random_state=0)
            forest.fit(X_train, y_train)

            # Get the variable importance using the test data
            result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
            forest_importances = pd.Series(result.importances_mean, index=feature_names)

            # Add into a DataFrame
            if isinstance(output, list):
                output = pd.DataFrame(
                    {'importances': forest_importances, 'importances_std': result.importances_std, 'country': country,
                     'gwl': gwl})
            else:
                newoutput = pd.DataFrame(
                    {'importances': forest_importances, 'importances_std': result.importances_std, 'country': country,
                     'gwl': gwl})
                output = pd.concat([output, newoutput])

    # Now convert the index into a column
    output = output.reset_index(inplace=False).rename(columns={'index': 'Variable'})
    # Make columns categorical to help with plotting
    output['Variable'] = pd.Categorical(output['Variable'], output['Variable'].unique())
    output['country'] = pd.Categorical(output['country'], output['country'].unique())
    output['gwl'] = pd.Categorical(output['gwl'], output['gwl'].unique())
    # Make variable names nicer for plotting
    lookup = {'air_temperature': 'T', 'precipitation_in_mpersec': 'Pr', 'relative_humidity': 'RH',
              'soil_moisture_content_top_metre': 'SM', 'windspeed_in_km_per_hr': 'W'}
    output['Variable_nice'] = 'T'
    for k, v in lookup.items():
        output['Variable_nice'].loc[output['Variable'] == k] = v

    return output


def sample_and_varimp(frac = 0.0001, nsamp=300, set=1, iterations=100):
    """
    Wrapper function to sample the big data and get variable importance
    :param frac: fraction to sample
    :param nsamp: number of samples per country per GWL
    :param set: set number (for multiple sampling sets)
    :param iterations: number of iterations to do
    :return: big dataframe of variable importance and variances for plotting
    """

    dfout = pd.DataFrame()

    for iteration in range(1, iterations+1):
        outfile = f'/data/users/andrew.hartley/Projects/FFDI_Paper/data/Sampled/set{set}/variable_importance_ffdi_12plus_sampled-n{nsamp}_set-{set}_iteration-{iteration}.csv'
        if not os.path.isfile(outfile):
            print(set, frac, iteration)
            df = sample_data(frac=frac, nsamp=nsamp, set=set, iteration=iteration)
            varimp = get_variable_importance(df)
            varimp.to_csv(outfile)
            print('Saved ', outfile)
        else:
            varimp = pd.read_csv(outfile)

        dfout = pd.concat([dfout, varimp])

    return dfout


def mask2df(cube, mask_arr, varname, gwlname, ensmem, rcp, region_name='natural_earth_v5_0_0.countries_110', country_list=['Australia', 'United States of America', 'Brazil']):
    """
    Mask the cube data using the mask, and creates a pandas dataframe using the additional metadata
    :param cube: 3D cube containing the data to be extracted and masked
    :param mask_arr: 3D numpy.ma mask (assumes mask=False for cells we want to keep)
    :param varname: variable name
    :param gwlname: global warming level name
    :param ensmem: ensemble members name
    :param rcp: RCP name
    :param region_name: could be 'ipcc' or anything accepted by regionmask. Options include all those available in regionmask: ['ar6.all', 'ar6.land', 'ar6.ocean', 'natural_earth_v5_0_0.countries_110', 'natural_earth_v5_0_0.countries_50', 'natural_earth_v5_0_0.countries_10','natural_earth_v5_0_0.land_110', 'natural_earth_v5_0_0.ocean_basins_50']
    :param country_list: list of countries to extract
    :return: pandas dataframe
    """
    import regionmask

    cells_we_want = ~mask_arr
    value = ma.getdata(cube.data)[cells_we_want].flatten()

    myu = cube.coord('time').units
    t = cube.coord('time').points
    y = cube.coord('latitude').points
    x = cube.coord('longitude').points
    ts, ys, xs = np.meshgrid(t, y, x, sparse=False, indexing='ij')

    ts_out = ts[cells_we_want].flatten()
    xs_out = [((x + 180) % 360) - 180 for x in xs[cells_we_want].flatten()]  # Corrects for 0 to 360 problem
    ys_out = ys[cells_we_want].flatten()
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=xs_out, y=ys_out), crs="EPSG:4326")

    if region_name == 'ipcc':
        inshp = '~/PycharmProjects/impactstoolbox/impactstoolbox/data/IPCC-WGI-reference-regions-v4.shp'
        shp = gpd.read_file(inshp, crs="epsg:4326")
    else:
        # region_name = 'natural_earth_v5_0_0.countries_110'
        regions = getattr(getattr(regionmask.defined_regions, region_name.split('.')[0]), region_name.split('.')[1])
        shp = regions.to_geodataframe()
        shp = shp.set_crs(epsg=4326, allow_override=True)
        shp = shp[shp['names'].isin(country_list)]

    sj = points.sjoin(shp, how="left")
    sj.reset_index(inplace=True)
    sj.drop_duplicates(subset=['index', 'geometry'], keep='first', inplace=True)

    # The most likely cause of problems could be due to the joined data not having the same number of records as
    # other records
    # print(len(xs_out))
    # print(len(sj['Acronym'].to_list()))
    if region_name == 'ipcc':
        dfout = pd.DataFrame({'t': myu.num2date(ts_out), 'x': xs_out, 'y': ys_out, 'value': value, 'var': varname,
                              'names': sj['Acronym'].to_list(), 'gwl': gwlname, 'ensemble_member': ensmem, 'rcp': rcp})  #
    else:
        dfout = pd.DataFrame({'t': myu.num2date(ts_out), 'x': xs_out, 'y': ys_out, 'value': value, 'var': varname,
                              'names': sj['names'].to_list(), 'gwl': gwlname, 'ensemble_member': ensmem,
                              'rcp': rcp})  #

    return dfout


def get_data_to_plot(type, bl_start_yr, df, climatology=True):
    '''

    :param type: Could be any one of Baseline, GWL1.5, GWL2, GWL4, GWL1.5_change, GWL2_change, GWL4_change
    :param bl_start_yr: baseline start year usually 1986
    :param df: Simple dataframe of the start year for each GWL and hist & fut jobid
    :param climatology: boolean to say whether to create a climatology for each grid cell,
    or to just export to csv the full daily data (useful for correlation plots)
    :return:
    '''

    # Root dir of the data
    indir = '/data/users/andrew.hartley/Projects/FFDI_Paper/ffdi_input_variables/'
    # indir = '/scratch/hadin/fire/ffdi_inputs/'

    # Create an empty variables
    country_cube = None
    hist_cubes = []
    fut_cubes = []

    # GWL LUT
    gwl_lut = {'GWL1.5': 'gwl_1_5_start',
               'GWL2': 'gwl_2_start',
               'GWL4': 'gwl_4_start'}

    # Make empty output dataframe
    column_names = ['Date', 'Ensemble_Member', 'Country', 'GWL', 'FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']
    odf = pd.DataFrame(columns=column_names)

    # Loop through each row of the table (contains the RCP & future jobid)
    for i, row in df.iterrows():
        print(i, row['rcp'], row['hist'], row['fut'])
        rcp = row['rcp'].replace('.', '_')

        ##########
        # Get the pre-calculated FFDI (saved as a timeseries of baseline + future, based on future ens_member_id
        print('   Loading FFDI ... ', end='')
        ffdi_file = '/data/users/andrew.hartley/Projects/FFDI_Paper/ffdi_output_joined/'+rcp+'/ffdi_joined_'+rcp+'_'+row['fut']+'.nc'
        print(ffdi_file)
        ffdi_cube = iris.load_cube(ffdi_file) # baseline and futurev
        # Time subset of ffdi
        # NB: Even for future plots, we still need the baseline FFDI >= 24 mask
        hist_start = cftime.Datetime360Day(bl_start_yr, 1, 1)
        hist_end = cftime.Datetime360Day(bl_start_yr + 20, 1, 1)
        hist_ffdi = periodConstraint(ffdi_cube, hist_start, hist_end)
        ffdi_vh_count = hist_ffdi.collapsed('time', iris.analysis.COUNT, function=lambda values: values >= 24)
        print('done')

        ##########
        # Get a gridded map of the country mask
        if country_cube is None:
            print('   Loading Country Mask ... ', end='')
            country_cube = get_country_shapefile(ffdi_vh_count)
            print('done')

        ##########
        # Get the data requested
        if type == 'Baseline' or 'change' in type:
            print('   Loading historical cubes ... ')
            hist_files = sorted(glob.glob(indir + rcp + '/*' + row['hist'] + '*'))
            hist_cubes = load_files(hist_files, bl_start_yr)
            print('   done')

        if 'GWL' in type:
            print('   Loading future cubes ... ')
            # Get the column name for this GWL, and check that it is not NA
            col = gwl_lut[type.split('_')[0]]
            if np.isnan(row[col]):
                # Means that this ens member didn't reach this GWL, so move to the next ens member
                continue
            fut_files = sorted(glob.glob(indir + rcp + '/*' + row['fut'] + '*'))
            fut_cubes = load_files(fut_files, row[col])
            # Time subset of ffdi
            fut_start = cftime.Datetime360Day(row[col], 1, 1)
            fut_end = cftime.Datetime360Day(row[col] + 20, 1, 1)
            fut_ffdi = periodConstraint(ffdi_cube, fut_start, fut_end)
            print('   done')

        ##########
        # Next, do the masking of FFDI, Countries and extract into a dataframe
        if hist_cubes and fut_cubes:
            # Ignore for the moment
            # need to calculate change in some way
            # Maybe per pixel climatologies of seasons DJF, MAM, JJA, SON?
            # Maybe annual means per pixel
            print('Nothing to do here yet')

        elif hist_cubes and not fut_cubes:
            print('   Masking & extracting historical cubes ... ')
            # Baseline only
            odf = mask_and_extract(hist_cubes, hist_ffdi, ffdi_vh_count, country_cube, row['fut'], type, row['rcp'], odf)
            print('   done')

        elif not hist_cubes and fut_cubes:
            print('   Masking & extracting future cubes ... ')
            # Future absolute
            odf = mask_and_extract(fut_cubes, fut_ffdi, ffdi_vh_count, country_cube, row['fut'], type, row['rcp'], odf)
            print('   done')

        else:
            # nothing in hist_cubes or fut_cubes, so there's a problem
            return None

        # Write out temporary odf file for testing plotting
        try:
            odf.drop(columns='Unnamed: 0', inplace=True)
        except:
            print('Don\'t worry about it')

        # pdb.set_trace()

        if climatology:
            # TODO: Check what this line does, and does it introduce any inconsistencies esp for GWL1.5
            cell_climatologies = odf.groupby(['CellX', 'CellY', 'Ensemble_Member', 'Country', 'GWL', 'RCP']).agg(
                {'FFDI': 'mean', 'RH': 'mean', 'Temperature': 'mean', 'Precipitation': 'mean', 'Wind_Speed': 'mean',
                 'Soil_Moisture': 'mean'})
            cell_climatologies.reset_index(inplace=True)
            cell_climatologies.to_csv('/data/users/andrew.hartley/Projects/FFDI_Paper/odf_climatology_'+str(i)+'_'+type+'.csv', index=False)
        else:
            odf.to_csv('/data/users/andrew.hartley/Projects/FFDI_Paper/odf_allcells_'+str(i)+'_'+type+'.csv', index=False)

    # return cell_climatologies


def mask_and_extract(cubelist, ffdi, ffdi_vh_count, country_cube, ens, type, rcp, odf, ffdi_threshold=20):
    '''
    Uses the ffdi_mask and country_cube to mask data from cubelist and
        puts it into a dataframe that can be used for plotting
    :param cubelist: Each cube contains the variables needed to calculate the FFDI
    :param ffdi: Daily FFDI during the period
    :param ffdi_vh_count: Count of grid cells with >1 day per year Very High FFDI in the baseline period, to use as a mask
    :param country_cube: Grid of countries (Australia, Brazil and USA) for masking
    :param ens: ensemble member
    :param rcp: rcp
    :param type: baseline, GWL1.5, GWL2, GWL4, GWL1.5_change, GWL2_change, or GWL4_change
    :param odf: Big dataframe for plotting. Columns = ['Ensemble_Member', 'Country', 'GWL', 'FFDI', 'RH', 'Temperature',
       'Precipitation', 'Wind_Speed', 'Soil_Moisture']
    :return: pandas dataframe for plotting
    '''


    col_lut = {'relative_humidity': 'RH', 'air_temperature': 'Temperature', 'precipitation_in_mpersec': 'Precipitation', 'windspeed_in_km_per_hr': 'Wind_Speed', 'soil_moisture_content_top_metre': 'Soil_Moisture'}

    for k, v in country_cube.attributes.items():

        # Mask using FFDI timeseries
        print('   ... Applying FFDI and Country mask ... ' + k)

        columns = ['Ensemble_Member', 'Country', 'GWL', 'variable', 'value', 'Date', 'RCP', 'CellX', 'CellY']
        results = pd.DataFrame(columns=columns)

        ffdi_cntry = ffdi.copy()
        # Get 2D mask of cells with < 20 V. High FFDI days from baseline period for this country
        data2d = ma.masked_where((country_cube.data.data != v) | (ffdi.data.data < ffdi_threshold), ffdi.data.data)
        # Broadcast the 2D mask to the ffdi (daily) cube shape
        basemask = np.broadcast_to(data2d.mask, ffdi.shape)
        # Add to the mask the days in the timeseries when FFDI < 24
        newmask = ffdi_cntry.data.mask | basemask
        # Adds the mask to the cube
        ffdi_cntry.data.mask = newmask
        # Gets the values that are not masked
        vals = ma.compressed(ffdi_cntry.data)
        # Gets the time coordinates for each grid cell that is not masked
        myu = ffdi_cntry.coord('time').units
        dates = myu.num2date(ffdi_cntry.coord('time').points[np.where(ffdi_cntry.data.mask == False)[0]])
        # Gets the cell coordinates where we have data
        cellx, celly = np.where(ffdi_cntry.data.mask == False)[1:]
        # Puts the FFDI values in the results dataframe
        these_results = pd.DataFrame({columns[0]: ens,
                                      columns[1]: k,
                                      columns[2]: type,
                                      columns[3]: 'FFDI',
                                      columns[4]: vals,
                                      columns[5]: dates,
                                      columns[6]: rcp,
                                      columns[7]: cellx,
                                      columns[8]: celly})
        results = pd.concat([results, these_results])
        
        for cube in cubelist:

            cube.data.mask = newmask
            vals = ma.compressed(cube.data)
            myu = cube.coord('time').units
            dates = myu.num2date(cube.coord('time').points[np.where(cube.data.mask == False)[0]])
            cellx, celly = np.where(cube.data.mask == False)[1:]
            these_results = pd.DataFrame({columns[0]: ens,
                                          columns[1]: k,
                                          columns[2]: type,
                                          columns[3]: col_lut[cube.name()],
                                          columns[4]: vals,
                                          columns[5]: dates,
                                          columns[6]: rcp,
                                          columns[7]: cellx,
                                          columns[8]: celly})
            results = pd.concat([results, these_results])

        # Change the pandas DataFrame to wide format (i.e. variables are now columns)
        try:
            df_pivot = results.pivot_table(index=[columns[0], columns[1], columns[2], columns[5], columns[6], columns[7], columns[8]], columns='variable', values='value').reset_index()
            odf = pd.concat([odf, df_pivot])
        except:
            print('Unable to add results to odf')

    return odf


def get_country_shapefile(cube):
    '''
    Get Country Masks (NB: has to switch to -180 to 180, and then back to 0 to 360)
    USA=4, BRA=5, AUS=1
    :param cube: iris.cube.Cube - has to be 2D (longitude, latitude coordinates only)
    :return: cube with the countries burnt into it
    '''

    fire_paper_countries = 'data/fire_paper_countries.shp'

    country_cube = hf.poly2cube(fire_paper_countries, 'MAPCOLOR7', cube.intersection(longitude=(-180, 180)))
    country_cube = country_cube.intersection(longitude=(0, 360))
    country_cube.coord('longitude').circular = True
    country_cube.data = ma.masked_equal(country_cube.data, 0)
    country_cube.attributes = {'Australia': 1, 'USA': 4, 'Brazil': 5}

    return country_cube


def plot_gwl_year(ensdf):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    # sns.set_palette('bright')
    sns.set_palette(sns.color_palette('bright')[1:3])

    ensdf.columns = ['rcp', 'hist', 'fut', '1.5 Degrees', '2 Degrees', '4 Degrees']
    ensdflong = pd.melt(ensdf, id_vars=['rcp', 'hist', 'fut'], value_vars=['1.5 Degrees', '2 Degrees', '4 Degrees'])
    ensdflong['Median Year'] = ensdflong['value'] + 10
    ax = sns.boxplot(data=ensdflong, y='variable', x='Median Year', hue='rcp', whis=[0,100], boxprops=None)
    ax = sns.stripplot(data=ensdflong, y='variable', x='Median Year', hue='rcp', palette={'rcp2.6': 'Black', 'rcp8.5': 'black'}, dodge=True, size=3)
    ax.set(xlim=(2000, 2100))

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2])

    plt.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/gwls_by_rcp.png')


def plot_rh_vs_temp(df):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set_palette('bright')

    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    # Groups all grid cells of an ensemble member for a country, GWL (and implicitly RCP)
    tmp = df.groupby(['Country', 'Ensemble_Member', 'GWL', 'RCP']).agg({
        'Temperature': 'median',
        'RH': 'median',
        'FFDI': 'median',
        'Precipitation': 'median',
        'Wind_Speed': 'median',
        'Soil_Moisture': 'median' # percentile(90)
    })
    tmp.reset_index(inplace=True)

    # sizes = [23, 27, 33]
    # marker_size = pd.cut(tmp['FFDI'], [21, 25, 29, 37], labels=sizes)

    g = sns.relplot(data=tmp, x='Temperature', y='RH', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)", y_var="RH (%)")
    g.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/temp_vs_rh_scatter.png')

    g = sns.relplot(data=tmp, x='Temperature', y='Precipitation', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)", y_var="Precipitation (mm.day$^{-1}$)")
    g.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/temp_vs_precip_scatter.png')

    g = sns.relplot(data=tmp, x='Temperature', y='Soil_Moisture', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)", y_var="Soil Moisture (kg.m$^{-2}$)")
    g.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/temp_vs_soilmoisture_scatter.png')

    g = sns.relplot(data=tmp, x='Temperature', y='Wind_Speed', hue='GWL',  # size=marker_size, # size='FFDI',
        size='FFDI', sizes=(1,200), style='RCP', col='Country', alpha=.5, kind='scatter')
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels(x_var="Temperature ($^\circ$C)",  y_var="Wind Speed (m.s$^{-1}$)")
    g.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/temp_vs_wind_scatter.png')


def plot_cell_climatologies(df):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set_palette('bright')

    # With stripplot ...
    for var in ['FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']:
        print(var)
        sns_plot = sns.catplot(x='GWL', y=var, hue='RCP', data=df, col='Country', kind='box', whis=[0, 100])
        sns_plot = sns_plot.map(sns.stripplot, 'GWL', var, 'RCP', dodge=True,
                                palette={'Baseline': 'black', 'rcp2.6': 'Black', 'rcp8.5': 'black'}, size=1,
                                jitter=0.35, alpha=0.2)
        [plt.setp(ax.texts, text="") for ax in sns_plot.axes.flat]
        sns_plot.set_titles(col_template='{col_name}')
        sns_plot.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/figure_'+var+'.png')

    # Violin ...
    for var in ['FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']:
        print(var)
        sns_plot = sns.catplot(x='GWL', y=var, hue='RCP', data=df, col='Country', kind='violin', dodge=True)
        [plt.setp(ax.texts, text="") for ax in sns_plot.axes.flat]
        sns_plot.set_titles(col_template='{col_name}')
        sns_plot.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/figure_'+var+'_violin.png')

    # Without stripplot ...
    for var in ['FFDI', 'RH', 'Temperature', 'Precipitation', 'Wind_Speed', 'Soil_Moisture']:
        print(var)
        sns_plot = sns.catplot(x='GWL', y=var, hue='RCP', data=df, col='Country', kind='box', whis=[0, 100])
        [plt.setp(ax.texts, text="") for ax in sns_plot.axes.flat]
        sns_plot.set_titles(col_template='{col_name}')
        sns_plot.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/figure_'+var+'_nopoints.png')

    # change df to long format
    df.columns = ['CellX', 'CellY', 'Ensemble Member', 'Country', 'GWL', 'RCP', 'FFDI',
                  'Relative Humidity', 'Temperature', 'Precipitation', 'Wind Speed', 'Soil Moisture']
    dflong = pd.melt(df, id_vars=['CellX', 'CellY', 'Ensemble Member', 'Country', 'GWL', 'RCP'], value_vars=['FFDI',
       'Relative Humidity', 'Temperature', 'Precipitation', 'Wind Speed', 'Soil Moisture'])

    # All in one figure (points and boxplots)
    g = sns.FacetGrid(data=dflong, col='Country', row='variable', margin_titles=True, sharex=True, sharey='row')
    g.map(sns.boxplot, 'GWL', 'value', 'RCP', hue_order=['Baseline','rcp2.6','rcp8.5'], whis=[0,100], palette='bright')
    g.map(sns.stripplot, 'GWL', 'value', 'RCP', dodge=True, hue_order=['Baseline','rcp2.6','rcp8.5'],
                                palette={'Baseline': 'black', 'rcp2.6': 'Black', 'rcp8.5': 'black'}, size=0.5,
                                jitter=0.35, alpha=0.2)
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    g.fig.subplots_adjust(wspace=.02, hspace=.02)
    g.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/figure_allvars_withpoints.png')

    # All in one figure (boxplots only)
    p = sns.FacetGrid(data=dflong, col='Country', row='variable', margin_titles=True, sharex=True, sharey='row')
    p.map(sns.boxplot, 'GWL', 'value', 'RCP', hue_order=['Baseline', 'rcp2.6', 'rcp8.5'], whis=[0, 100],
          palette=sns.color_palette(['#99a3a4','#0097ff','#d7301f'])) # palette='bright'
    [plt.setp(ax.texts, text="") for ax in p.axes.flat]
    p.set_titles(row_template='{row_name}', col_template='{col_name}')
    ylabels = ["FFDI", "Relative Humidity (%)", "Temperature ($^\circ$C)", "Precipitation (mm.day$^{-1}$)", "Wind Speed (m.s$^{-1}$)", "Soil Moisture (kg.m$^{-2}$)"]
    for i in range(len(ylabels)):
        p.axes[i, 0].set_ylabel(ylabels[i])
    # p.add_legend(loc='center right', bbox_to_anchor=(0.87, 0.47), frameon=True)
    p.add_legend(loc='center left', bbox_to_anchor=(0.76, 0.47), frameon=True)
    p.fig.subplots_adjust(wspace=0, hspace=0)
    p.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/figure_allvars_nopoints.png')

    # All in one figure (boxplots only), RCP x-axis, GWL colours
    p = sns.FacetGrid(data=dflong, col='Country', row='variable', margin_titles=True, sharex=True, sharey='row')
    p.map(sns.boxplot, 'RCP', 'value', 'GWL', hue_order=['Baseline', 'GWL1.5', 'GWL2', 'GWL4'], whis=[0, 100],
          palette=sns.color_palette(['#99a3a4','#0097ff','#fc641a','#d7301f'])) # palette='bright'
    [plt.setp(ax.texts, text="") for ax in p.axes.flat]
    p.set_titles(row_template='{row_name}', col_template='{col_name}')
    ylabels = ["FFDI", "Relative Humidity (%)", "Temperature ($^\circ$C)", "Precipitation (mm.day$^{-1}$)", "Wind Speed (m.s$^{-1}$)", "Soil Moisture (kg.m$^{-2}$)"]
    for i in range(len(ylabels)):
        p.axes[i, 0].set_ylabel(ylabels[i])
    # p.add_legend(loc='center right', bbox_to_anchor=(0.87, 0.47), frameon=True)
    p.add_legend(loc='center left', bbox_to_anchor=(0.76, 0.47), frameon=True)
    p.fig.subplots_adjust(wspace=0, hspace=0)
    p.savefig('/data/users/andrew.hartley/Projects/FFDI_Paper/plots/figure_allvars_nopoints_RCP-xaxis.png')


def main_old():
    # This code produced plots with outliers in the variable relationships.
    # I think it was due to problems with the masking. I only looked at grid cells where FFDI>24 was recorded for a
    # certain number of days. This may have introduced some erros

    # Get the data from MASS. Possibly a problem withe the joined data, so using the above data instead
    if not os.path.isdir(f"{scratchdir}fire/ffdi_output_joined/rcp2_6/"):
        os.makedirs(f"{scratchdir}fire/ffdi_output_joined/rcp2_6/")
    if not os.path.isdir(f"{scratchdir}fire/ffdi_output_joined/rcp8_5/"):
        os.makedirs(f"{scratchdir}fire/ffdi_output_joined/rcp8_5/")

    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6_joined/*.nc", f"{scratchdir}fire/ffdi_output_joined/rcp2_6/"])
    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5_joined/*.nc", f"{scratchdir}fire/ffdi_output_joined/rcp8_5/"])

    # Getting FFDI input variables from MASS
    if not os.path.isdir(f"{scratchdir}fire/ffdi_input_variables/rcp2_6/"):
        os.makedirs(f"{scratchdir}fire/ffdi_input_variables/rcp2_6/")
    if not os.path.isdir(f"{scratchdir}fire/ffdi_input_variables/rcp8_5/"):
        os.makedirs(f"{scratchdir}fire/ffdi_input_variables/rcp8_5/")

    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6/ffdi_input_variables*.nc", f"{scratchdir}fire/ffdi_input_variables/rcp2_6/"])
    subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5/ffdi_input_variables*.nc", f"{scratchdir}fire/ffdi_input_variables/rcp8_5/"])

    # Load the ensemble jobids and GWLs
    ensdf = get_ens_df()
    plot_gwl_year(ensdf)

    # Baseline start year
    bl_start_yr = 1986

    # Get the baseline data to plot
    csv_bl = '/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_113_Baseline.csv'
    if not os.path.isfile(csv_bl):
        df2p_baseline = get_data_to_plot('Baseline', bl_start_yr, ensdf)
    else:
        df2p_baseline = pd.read_csv(csv_bl)
    df2p_base = df2p_baseline[df2p_baseline.RCP == 'rcp2.6'].copy()
    df2p_base['RCP'] = 'Baseline'

    # Get GWL1.5 data to plot
    csv_gwl1p5 = '/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_113_GWL1.5.csv'
    if not os.path.isfile(csv_gwl1p5):
        df2p_gwl1p5 = get_data_to_plot('GWL1.5', bl_start_yr, ensdf)
    else:
        df2p_gwl1p5 = pd.read_csv(csv_gwl1p5)

    # Get GWL2 data to plot
    csv_gwl2 = '/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_113_GWL2.csv'
    if not os.path.isfile(csv_gwl2):
        df2p_gwl2 = get_data_to_plot('GWL2', bl_start_yr, ensdf)
    else:
        df2p_gwl2 = pd.read_csv(csv_gwl2)

    # Get GWL4 data to plot
    csv_gwl4 = '/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_113_GWL4.csv'
    if not os.path.isfile(csv_gwl4):
        df2p_gwl4 = get_data_to_plot('GWL4', bl_start_yr, ensdf)
    else:
        df2p_gwl4 = pd.read_csv(csv_gwl4)

    df = pd.concat([df2p_base, df2p_gwl1p5, df2p_gwl2, df2p_gwl4], ignore_index=True)
    plot_cell_climatologies(df)

    df = pd.concat([df2p_baseline, df2p_gwl1p5, df2p_gwl2, df2p_gwl4], ignore_index=True)
    plot_rh_vs_temp(df)


def main():

    # Get the data from MASS. This is the actual data that we will use ...
    if not os.path.isdir(f"{scratchdir}fire/ffdi_output/rcp2_6/"):
        os.makedirs(f"{scratchdir}fire/ffdi_output/rcp2_6/")
    if not os.path.isdir(f"{scratchdir}fire/ffdi_output/rcp8_5/"):
        os.makedirs(f"{scratchdir}fire/ffdi_output/rcp8_5/")

    if len(os.listdir(f'{scratchdir}fire/ffdi_output/rcp2_6/')) < 228:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6/ffdi_outputs/*.nc", f"{scratchdir}fire/ffdi_output/rcp2_6/"])
    if len(os.listdir(f'{scratchdir}fire/ffdi_output/rcp8_5/')) < 228:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5/ffdi_outputs/*.nc", f"{scratchdir}fire/ffdi_output/rcp8_5/"])

    # Getting FFDI input variables from MASS
    if not os.path.isdir(f"{scratchdir}fire/ffdi_input_variables/rcp2_6/"):
        os.makedirs(f"{scratchdir}fire/ffdi_input_variables/rcp2_6/")
    if not os.path.isdir(f"{scratchdir}fire/ffdi_input_variables/rcp8_5/"):
        os.makedirs(f"{scratchdir}fire/ffdi_input_variables/rcp8_5/")

    if len(os.listdir(f'{scratchdir}fire/ffdi_input_variables/rcp2_6/')) < 798:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp2_6/ffdi_input_variables*.nc", f"{scratchdir}fire/ffdi_input_variables/rcp2_6/"])
    if len(os.listdir(f'{scratchdir}fire/ffdi_input_variables/rcp8_5/')) < 798:
        subprocess.run(["moo", "get", "-v", "moose:/adhoc/users/inika.taylor/fire/rcp8_5/ffdi_input_variables*.nc", f"{scratchdir}fire/ffdi_input_variables/rcp8_5/"])

    # Load the ensemble jobids and GWLs
    ensdf = get_ens_df()
    # ensdf_4c = ensdf[~ensdf.gwl_4_start.isnull()]
    ensdf_lastbit = ensdf.iloc[100:]
    # plot_gwl_year(ensdf)

    # Get all data to plot
    countries_3 = ['Australia', 'United States of America', 'Brazil']
    countries_4 = ['Australia', 'United States of America', 'Brazil', 'South Africa']
    ## 3 countries
    # Run this for the majority of plots ...
    # df_12_inf = get_variables_by_ffdi_theshold_nosample(ensdf_lastbit, 12, np.inf, ofile='/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_bigdata_3countries_12_to_inf.csv', country_list=countries_3)

    # df_12_24 = get_variables_by_ffdi_theshold(ensdf, 12, 24, ofile='/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_bigdata_3countries_12_to_24.csv', country_list=countries_3)
    # df_gt_24 = get_variables_by_ffdi_theshold(ensdf, 24, np.inf, ofile='/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_bigdata_3countries_gt_24.csv', country_list=countries_3)
    # ## 4 countries
    # df_12_24 = get_variables_by_ffdi_theshold(ensdf, 12, 24, ofile='/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_bigdata_4countries_12_to_24.csv', country_list=countries_4)
    # df_gt_24 = get_variables_by_ffdi_theshold(ensdf, 24, np.inf, ofile='/data/users/andrew.hartley/Projects/FFDI_Paper/data/odf_bigdata_4countries_gt_24.csv', country_list=countries_4)

    # Run this for sampling and testing variable importance
    # sample_and_varimp()
    sample_and_varimp(nsamp=300, set=2, iterations=1000)


if __name__ == '__main__':
    main()
