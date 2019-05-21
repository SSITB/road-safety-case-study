import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class CleanData():
    
    def __init__(self):
        self.X = self
    
    def merge_tables(self, acc, veh, cas):
    
        #Add age of driver and age of casualty
        temp1 = pd.DataFrame(veh.groupby('Accident_Index')[[
                                                    'Age_of_Driver',
                                                    'Age_of_Vehicle',
                                                    'Engine_Capacity_(CC)']].mean())
        temp2 = pd.DataFrame(cas.groupby('Accident_Index')['Age_of_Casualty'].mean())
    
        #Merge Ages of driver and casualty with Accidents dataset by Accident_Index
        acc1=pd.merge(acc, temp1, on='Accident_Index',how='outer')
        acc1=pd.merge(acc1, temp2, on='Accident_Index',how='outer')
    
        #Take only rows that contain no of casualties, age of driver and age of casualty
        acc2 = acc1.dropna(subset=['Number_of_Casualties', 
                                   'Age_of_Driver', 
                                   'Age_of_Casualty'])
        
        return acc2
    
    
    def missing_values(self, df):
        #Replacing missing values with the mean
        df['Speed_limit'] = df['Speed_limit'].replace(np.nan, df['Speed_limit'].mean())
    
        vars = ['Age_of_Driver', 'Age_of_Vehicle', 'Engine_Capacity_(CC)', 'Age_of_Casualty']
        for i in vars:
            df[i]=df[i].replace(-1, df[i].mean())
        return df
    
    
    def features_gbr(self, df):
        
        vars = ['TimeH']
        for i in vars:
            df[i]=df[i].replace(np.nan, df[i].mean())
            
        X = df[['Number_of_Vehicles', 'Police_Force', 'Accident_Severity', 'Day_of_Week',
                'Local_Authority_(District)', '1st_Road_Class', 'Road_Type','Junction_Detail',
                'Speed_limit', 'TimeH', 'Junction_Control', '2nd_Road_Class', 
                'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities',
                'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 
                'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Urban_or_Rural_Area',
                'Did_Police_Officer_Attend_Scene_of_Accident','Age_of_Driver',
                'Age_of_Casualty','Number_of_Casualties','Age_of_Vehicle',
                'Engine_Capacity_(CC)']]
        
        X['Number_of_Vehicles'] = np.log(X['Number_of_Vehicles'])
        
        return X


    
    def features(self, df):
        
        X = df[['Age_of_Driver', 'Age_of_Casualty', 'Number_of_Casualties',
                'Number_of_Vehicles', 'Age_of_Vehicle', 'Engine_Capacity_(CC)', 
                'Speed_limit']]
        
        # Create time variable
        df.Time = pd.to_datetime(df.Time, format='%H:%M')
        X['TimeH'] = pd.DatetimeIndex(df.Time).hour
        
        # Create square terms of age variables
        X[['Age_of_Driver_sq','Age_of_Casualty_sq','Age_of_Vehicle_dq']] = df[[
                            'Age_of_Driver','Age_of_Casualty','Age_of_Vehicle']]**2
                
        # Take logs of highly skewed features
        X[['Number_of_Vehicles','Engine_Capacity_(CC)']] = np.log(df[[
            'Number_of_Vehicles','Engine_Capacity_(CC)']])
        
        # =============================================================================
        # Dummy variables
        # =============================================================================
       
        # Dummy for Policy force - Metropolitan
        X['Police_Force_metropolitan']=df['Police_Force'].where(
                                            df['Police_Force']==1.0,0) 
        
        # Dummies for accident severity
        X[['Acc_severity_fatal','Acc_severity_serious']] = pd.get_dummies(
                    df['Accident_Severity'].map({1: 'Acc_severity_fatal', 
                                                2: 'Acc_severity_serious'}))
    
        # Dummies for day of the week, Monday is omited
        X[['Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday', 'Sunday'
            ]] = pd.get_dummies(df['Day_of_Week'].map({2: 'Tuesday',
                                                        3: 'Wednesday',
                                                        4: 'Thursday',
                                                        5: 'Friday',
                                                        6: 'Saturday',
                                                        7: 'Sunday'}))
        
        # Dummies for 5 districts with most accidents
        districts = [300, 204, 1, 926, 200]
        districts_names = ['Birmingham', 'Leeds', 'Westminister', 'Glasgow_city','Bradford']
        for i in range(len(districts)):
            X.loc[:, districts_names[i]]=df['Local_Authority_(District)'].where(
                                            df['Local_Authority_(District)']==districts[i],0)
            X.loc[:, districts_names[i]] = X.loc[:, districts_names[i]].where(
                                            X.loc[:, districts_names[i]]==0,1)
    
    
        # Dummies for 5 highway authorities with most accidents
        highway = ['E10000016', 'E10000030', 'E10000017', 'E10000012', 'E08000025']
        highway_names = ['Kent', 'Surrey', 'Lancashire', 'Essex','Birmingham']
    
        for i in range(len(highway)):
            X.loc[:,highway_names[i]]=df['Local_Authority_(Highway)'].where(
                                            df['Local_Authority_(Highway)']==highway[i],0)
            X.loc[:,highway_names[i]] = X.loc[:,highway_names[i]].where(
                                            X.loc[:,highway_names[i]]==0,1)
            
            
        # Dummy for 1st Road Class
        dummies_1stroadclass = pd.get_dummies(df['1st_Road_Class'])
        dummies_1stroadclass = dummies_1stroadclass.add_suffix('_1st_road_class')
        X = pd.concat([dummies_1stroadclass['1.0_1st_road_class'], X], axis=1, 
                                                              join_axes=[X.index])    
        
        # Dummy for 1st Road Class
        dummies_roadtype = pd.get_dummies(df['Road_Type'], drop_first=True)
        dummies_roadtype = dummies_roadtype.add_suffix('_roadtype')
        X = pd.concat([dummies_roadtype['1.0_roadtype'], X], axis=1, join_axes=[X.index])    
        
        # Dummy for Junction detail
        dummies_junction_detail = pd.get_dummies(df['Junction_Detail'], drop_first=True)
        dummies_junction_detail = dummies_junction_detail.add_suffix('_junction_detail')
        X = pd.concat([dummies_junction_detail, X], axis=1, join_axes=[X.index])    
        X = X.drop('0.0_junction_detail', axis=1)
        
        # Dummy for Junction control
        dummies_junction_control = pd.get_dummies(df['Junction_Control'])
        dummies_junction_control = dummies_junction_control.add_suffix('_junction_control')
        X = pd.concat([dummies_junction_control['1.0_junction_control'], X], axis=1, 
                                                                      join_axes=[X.index])    
        
        # Dummy for 2nd Road Class
        dummies_2ndroadclass = pd.get_dummies(df['2nd_Road_Class'])
        dummies_2ndroadclass = dummies_2ndroadclass.add_suffix('_2nd_road_class')
        X = pd.concat([dummies_2ndroadclass['1.0_2nd_road_class'], X], axis=1, 
                                                                      join_axes=[X.index])    
        
        # Dummy for the presence of Human_Control in Pedestrian_Crossing
        X['Ped_Cross_None_Human_control']=df['Pedestrian_Crossing-Human_Control'].where(
                                                df['Pedestrian_Crossing-Human_Control']!=0.0,1) 
        X['Ped_Cross_None_Human_control']=X['Ped_Cross_None_Human_control'].where(
                                                X['Ped_Cross_None_Human_control']==1,0) 
        
        # Dummy for the presence of Pedestrial Crossing
        X['Ped_Cross_No_cross_facility']=df['Pedestrian_Crossing-Physical_Facilities'].where(
                df['Pedestrian_Crossing-Physical_Facilities']!=0.0,1) 
        X['Ped_Cross_No_cross_facility']=X['Ped_Cross_No_cross_facility'].where(
                X['Ped_Cross_No_cross_facility']==1,0) 
        
        # Dummy for 'Darkness - no lighting' light condition
        dummies_light = pd.get_dummies(df['Light_Conditions'], drop_first=True)
        dummies_light = dummies_light.add_suffix('_light')
        X = pd.concat([dummies_light['6.0_light'], X], axis=1, join_axes=[X.index])    
        
        # Dummies for weather condition
        dummies_weather = pd.get_dummies(df['Weather_Conditions'], drop_first=True)
        dummies_weather = dummies_weather.add_suffix('_weather')
        X = pd.concat([dummies_weather, X], axis=1, join_axes=[X.index])    
        X = X.drop('1.0_weather', axis=1)
        
        # Dummies for dry road surface condition
        dummies_road_surface = pd.get_dummies(df['Road_Surface_Conditions'], drop_first=True)
        dummies_road_surface = dummies_road_surface.add_suffix('_road_surface')
        X = pd.concat([dummies_road_surface['1.0_road_surface'], X], axis=1, join_axes=[X.index])
        
        
        # Dummy for the presence of special conditions at sight
        X['Special_Cond_at_Site_1']=df['Special_Conditions_at_Site'].where(
                                            df['Special_Conditions_at_Site']==0.0,1) 
     
        # Dummy for the presence of Carriageway Hazards
        X['Carriageway_Hazards_1']=df['Carriageway_Hazards'].where(
                                                df['Carriageway_Hazards']==0.0,1) 
        
        #Dummy for urban area
        X['Urban_area']=df['Urban_or_Rural_Area'].where(
                                                df['Urban_or_Rural_Area']==1,0)
        
        #Dummy = 1 if police officer attended the scene of accident
        X['Police_attend_1']=df['Did_Police_Officer_Attend_Scene_of_Accident'].where(
                df['Did_Police_Officer_Attend_Scene_of_Accident']==1,0)
        
        return X



