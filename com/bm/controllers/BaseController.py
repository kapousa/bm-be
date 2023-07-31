import logging
import os
import shutil
from datetime import datetime

import nltk
import numpy
import numpy as np
import pandas as pd
from flask import abort, session
from nltk.corpus import wordnet

from app import db
from app.modules.base.constants.BM_CONSTANTS import scalars_location, pkls_location, output_docs_location, df_location, \
    plot_zip_locations, plot_locations, html_plots_location, prediction_model_keyword, classification_model_keyword, \
    forecasting_model_keyword, clustering_model_keyword, \
    dep_path
from app.modules.base.db_models.ModelAPIDetails import ModelAPIDetails
from app.modules.base.db_models.ModelAPIModelMethods import ModelAPIModelMethods
from app.modules.base.db_models.ModelBotKeywords import ModelBotKeywords
from app.modules.base.db_models.ModelCvisionRun import ModelCvisionRun
from app.modules.base.db_models.ModelEncodedColumns import ModelEncodedColumns
from app.modules.base.db_models.ModelFeatures import ModelFeatures
from app.modules.base.db_models.ModelForecastingResults import ModelForecastingResults
from app.modules.base.db_models.ModelLabels import ModelLabels
from app.modules.base.db_models.ModelProfile import ModelProfile
from app.modules.base.constants.BM_CONSTANTS import deployment_folder
from com.bm.controllers.ControllersHelper import ControllersHelper
from com.bm.db_helper import AttributesHelper
from com.bm.utiles.Helper import Helper


class BaseController:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def deletemodels(self):
        try:
            all_models = self.get_all_models()

            # Delete old model files
            for model_profile in all_models:
                delete_model_files = self.deletemodel(model_profile['model_id'])

            return 1
        except Exception as e:
            print('Ohh -delet_models...Something went wrong.')
            print(e)
            return 0

    def deletemodel(self, model_id):
        try:
            ModelEncodedColumns.query.filter_by(model_id=model_id).delete()
            ModelFeatures.query.filter_by(model_id=model_id).delete()
            ModelLabels.query.filter_by(model_id=model_id).delete()
            ModelAPIModelMethods.query.filter_by(model_id=model_id).delete()
            ModelAPIDetails.query.filter_by(model_id=model_id).delete()
            ModelProfile.query.filter_by(model_id=model_id).delete()
            ModelForecastingResults.query.filter_by(model_id=model_id).delete()
            ModelCvisionRun.query.filter_by(model_id=model_id).delete()
            db.session.commit()

            # Delete all added files and folders
            datafilepath = "%s%s%s" % (df_location, str(model_id), '.csv')
            if os.path.exists(datafilepath):  # This check added to handle object detection models
                deletedatafile = os.remove(datafilepath)

            paths = {
                'ploting_path': html_plots_location + str(model_id),  # all geenrated html files
                'zip_path': plot_zip_locations + str(model_id),  # all generated iamge files
                'pkl_location': pkls_location + str(model_id),  # pkls location
                'output_document': output_docs_location + str(model_id),  # Output documents
                'model_data_location': df_location + str(model_id),  # model data location
                'plots_image_path': os.path.join(plot_locations, str(model_id)),  # plot images location
                'scalar_location': scalars_location + str(model_id),  # scalrs location
                'deployed_location': dep_path + str(model_id)
            }
            deletefolderfiles = Helper.deletefolderfiles(*paths.values())
            deleteobjdecfiles = Helper.deleteobjectdetectionfiles(model_id)

            for path in paths.values():  # Delete old folders
                shutil.rmtree(path) if (os.path.isdir(path)) else print(0)

            return 1

        except Exception as e:
            print('Ohh -delete_model...Something went wrong.' + e)
            print(e)
            return 0

    @staticmethod
    def get_cm_accurcy(c_m):
        print(c_m)
        sum_of_all_values = 0
        number_of_columns = len(c_m[0])
        correct_pre_sum = 0

        for i in range(number_of_columns):
            correct_pre_sum = correct_pre_sum + c_m[i][i]

        c_m = c_m.flatten()
        for i in range(len(c_m)):
            sum_of_all_values = sum_of_all_values + c_m[i]

        c_m_accurcy = round((correct_pre_sum / sum_of_all_values), 3) * 100

        return c_m_accurcy

    @staticmethod
    def get_all_models():
        try:
            model_profiles = ModelProfile.query.filter_by(user_id = session['logger']).order_by("updated_on").all()
            profiles = []

            for profile in model_profiles:
                model_profile = {'model_id': profile.model_id,
                                 'model_name': profile.model_name,
                                 'model_description': profile.description,
                                 'status': AttributesHelper.get_lookup_value(profile.status),
                                 'updated_on': profile.updated_on,
                                 'updated_by': AttributesHelper.get_user_fullname(profile.user_id),
                                 'ds_goal': profile.ds_goal,
                                 'deployed': AttributesHelper.get_lookup_value(profile.deployed),
                                 'Root_Mean_Squared_Error': profile.root_mean_squared_error,
                                 'Mean_Squared_Error': profile.mean_squared_error,
                                 'Accuracy': profile.accuracy
                                 }
                profiles.append(model_profile)

            return profiles
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return 0

    @staticmethod
    def get_model_status(model_id):
        try:
            model_profile_row = [ModelProfile.query.filter_by(model_id=model_id).first()]
            model_profile = {}

            for profile in model_profile_row:
                model_profile = {'model_id': profile.model_id,
                                 'model_name': profile.model_name,
                                 'prediction_results_accuracy': str(profile.prediction_results_accuracy),
                                 'mean_absolute_error': str(profile.mean_absolute_error),
                                 'mean_squared_error': str(profile.mean_squared_error),
                                 'root_mean_squared_error': str(profile.root_mean_squared_error),
                                 'plot_image_path': profile.plot_image_path,
                                 'created_on': profile.created_on,
                                 'updated_on': profile.updated_on,
                                 'last_run_time': profile.last_run_time,
                                 'ds_source': profile.ds_source,
                                 'ds_goal': profile.ds_goal,
                                 'mean_percentage_error': profile.mean_percentage_error,
                                 'mean_absolute_percentage_error': profile.mean_absolute_percentage_error,
                                 'depended_factor': profile.depended_factor,
                                 'forecasting_category': profile.forecasting_category,
                                 'train_precision': profile.train_precision,
                                 'train_recall': profile.test_f1,
                                 'train_f1': profile.test_f1,
                                 'test_precision': profile.test_f1,
                                 'test_recall': profile.test_f1,
                                 'test_f1': profile.test_f1,
                                 'description': profile.description,
                                 'status': profile.status,
                                 'deployed': profile.deployed,
                                 'running_duration': profile.running_duration,
                                 'accuracy': profile.accuracy,
                                 'classification_categories': profile.classification_categories,
                                 'most_common': profile.most_common
                                 }
            return model_profile
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return 0

    @staticmethod
    def initiate_model(model_id):

        paths = {
            'ploting_path': html_plots_location + str(model_id),  # all geenrated html files
            'zip_path': plot_zip_locations + str(model_id),  # all generated iamge files
            'pkl_location': pkls_location + str(model_id),  # pkls location
            'output_document': output_docs_location + str(model_id),  # Output documents
            'data_location': df_location + str(model_id),  # data location
            'plots_image_path': os.path.join(plot_locations, str(model_id)),  # plot images location
            'scalar_location': scalars_location + str(model_id),  # scalrs location
            'deployed_location': dep_path + str(model_id)
        }

        # Delet old folders and create new
        for path in paths.values():
            shutil.rmtree(path) if (os.path.isdir(path)) else os.mkdir(path)

        return 0

    def detectefittedmodels(self, user_desc):
        try:
            # Analysing the input
            text = nltk.word_tokenize(user_desc)
            pos_tagged = nltk.pos_tag(text)
            text_verbs = list(filter(lambda x: x[0], pos_tagged))
            text_verbs = numpy.array(text_verbs)
            text_verbs = text_verbs[:, 0] if len(text_verbs) > 0 else text_verbs

            if (len(text_verbs) == 0):
                return ["Sorry but we couldn't recognise what you need, Please rephrase your description and try again"]

            # Extract verbs
            synonyms_verbs = []
            for i in range(len(text_verbs)):
                xx = text_verbs[i]
                for syn in wordnet.synsets(text_verbs[i]):
                    for lm in syn.lemmas():
                        synonyms_verbs.append(lm.name())  # adding into synonyms
                print(set(synonyms_verbs))

            synonyms = synonyms_verbs
            synonyms = numpy.unique(synonyms)
            modelbotkeywords = ModelBotKeywords.query.all()
            suggested_models = []
            suggested_models_ids = []

            for i in range(len(synonyms)):
                for item in modelbotkeywords:
                    kwords = item.keywords
                    kwords = kwords.split(',')
                    # kwords = numpy.array(kwords)
                    if any(word.startswith(synonyms[i]) for word in kwords):
                        suggested_models.append(item.model_type)
                        suggested_models_ids.append(item.model_code)

            suggested_models = np.unique(suggested_models)
            # suggested_models = ''.join(suggested_models)

            if (len(suggested_models) == 0):
                return ["Sorry but we couldn't recognise what you need, Please rephrase your description and try again"]

            results = ['Well, Here how we can help you:']
            results.append(
                "- Create %s model to predict values based on the history of the old data.<br/><a href='/createmodel?t=7' class='btn btn-primary' style='float: right'>Create prediction model</a>" % (
                    prediction_model_keyword)) if (prediction_model_keyword in suggested_models) else print('0')
            results.append(
                "- Group data under sets of %s that help reaching to the data easily in the future.<br/><a href='/createmodel?t=10' class='btn btn-danger' style='float: right'>Connect to labeled data</a>" % (
                    classification_model_keyword)) if (classification_model_keyword in suggested_models) else print('0')
            results.append(
                "- Create %s model to predict values in specific Date/Time.<br/><a href='/createmodel?t=8' class='btn btn-default' style='float: right'>Start forecasting model</a>" % (
                    forecasting_model_keyword)) if (forecasting_model_keyword in suggested_models) else print('0')
            results.append(
                "- %s related information under one umbrella to make it easy for you to find information that have same characteristc.<br/><a href='/createmodel?t=13' class='btn btn-success' style='float: right'>Connect to row data</a>" % (
                    clustering_model_keyword)) if (clustering_model_keyword in suggested_models) else print('0')

            return results if len(results) > 1 else [
                "Sorry but we couldn't recognise what you need, Please rephrase your description and try again"]

        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return ['Ohh -get_model_status...Something went wrong.']

    def detectefittedmodels_(self, user_desc):
        try:
            # Analysing the input
            text = nltk.word_tokenize(user_desc)
            pos_tagged = nltk.pos_tag(text)
            text_verbs = list(filter(lambda x: x[1] == 'VB', pos_tagged))
            text_words = list(filter(lambda x: x[1] == 'NN', pos_tagged))
            text_verbs = numpy.array(text_verbs)
            text_verbs = text_verbs[:, 0] if len(text_verbs) > 0 else text_verbs
            text_words = numpy.array(text_words)
            text_words = text_words[:, 0] if len(text_words) > 0 else text_words

            if (len(text_verbs) == 0 and len(text_words) == 0):
                return ["Sorry but we couldn't recognise what you need, Please rephrase your description and try again"]

            # Extract verbs
            synonyms_verbs = []
            for i in range(len(text_verbs)):
                xx = text_verbs[i]
                for syn in wordnet.synsets(text_verbs[i]):
                    for lm in syn.lemmas():
                        synonyms_verbs.append(lm.name())  # adding into synonyms
                print(set(synonyms_verbs))

            # Extract words
            synonyms_text = []
            for i in range(len(text_words)):
                xx = text_words[i]
                for syn in wordnet.synsets(text_words[i]):
                    for lm in syn.lemmas():
                        synonyms_text.append(lm.name())  # adding into synonyms
                print(set(synonyms_text))

            synonyms = numpy.concatenate((synonyms_verbs, synonyms_text), axis=0)
            synonyms = numpy.unique(synonyms)
            modelbotkeywords = ModelBotKeywords.query.all()
            suggested_models = []
            suggested_models_ids = []

            for i in range(len(synonyms)):
                for item in modelbotkeywords:
                    kwords = item.keywords
                    kwords = kwords.split(',')
                    # kwords = numpy.array(kwords)
                    if any(word.startswith(synonyms[i]) for word in kwords):
                        suggested_models.append(item.model_type)
                        suggested_models_ids.append(item.model_code)

            suggested_models = np.unique(suggested_models)
            # suggested_models = ''.join(suggested_models)

            if (len(suggested_models) == 0):
                return ["Sorry but we couldn't recognise what you need, Please rephrase your description and try again"]

            results = ['Well, Here how we can help you:']
            results.append("- Create %s model to predict values based on the history of the old data." % (
                prediction_model_keyword)) if (prediction_model_keyword in suggested_models) else print('0')
            results.append(
                "- Group your data under set of %s to help you to reach to the data easily in the future." % (
                    classification_model_keyword)) if (classification_model_keyword in suggested_models) else print('0')
            results.append(
                "- Create %s model to predict values in specific Date/Time." % (forecasting_model_keyword)) if (
                        forecasting_model_keyword in suggested_models) else print('0')
            results.append(
                "- %s related information under one umbrella to make it easy for you to find information that have same characteristc." % (
                    clustering_model_keyword)) if (clustering_model_keyword in suggested_models) else print('0')

            sample_data = [
                              results.to_html(border=0, classes='table table-hover', header="false",
                                              justify="center").replace("<th>",
                                                                        "<th class='text-warning'>")],
            return sample_data

        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return ['Ohh -get_model_status...Something went wrong.']

    def updatemodelinfo(self, model_id, updated_model_name, updated_model_description):
        try:
            now = datetime.now()
            model_profile = ModelProfile.query.filter_by(model_id=model_id).first()
            model_profile.model_name = updated_model_name
            model_profile.description = updated_model_description
            model_profile.updated_on = now.strftime("%d/%m/%Y %H:%M:%S")
            db.session.commit()

            return 'Success'

        except  Exception as e:
            return 'Ohh -updatemodelinfo...Something went wrong.'

    def changemodelstatus(self, model_id):
        try:
            model_profile = ModelProfile.query.filter_by(model_id=model_id).first()
            model_profile.status = 20 if (model_profile.status == 19) else 19
            db.session.commit()

            return 'Success'

        except  Exception as e:
            return 'Ohh -suspendmodel...Something went wrong.'

    def deploymodel(self, model_id):
        try:
            if (not ControllersHelper.model_deployed(model_id)):
                try:
                    # 1- Copy model file
                    str_model_id = str(model_id)
                    path = os.path.join(deployment_folder, model_id)
                    source = "%s%s%s%s%s" % (pkls_location, str_model_id, '/', str_model_id, '_model.pkl')
                    destination = "%s%s%s%s%s" % (deployment_folder, str_model_id, '/', str_model_id, '_model.pkl')

                    # if(os.path.exists(destination)):
                    #     return "Model already deployed"
                    # else:
                    os.mkdir(path)
                    shutil.copy(source, destination)
                    print("File copied successfully.")

                    # 2- Update deployment status
                    model_profile = ModelProfile.query.filter_by(model_id=model_id).first()
                    model_profile.deployed = 23
                    db.session.commit()

                    return "Model deployed successfully"
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
                    return "Model already deployed"

                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")
                    return "There is no permission to deploy the model"

            else:
                return self.undeploymodel(model_id)

        # For other errors
        except Exception as e:
            print(e)
            print("Error occurred while copying file.")
            return "Failed to deploy the model, please try again or contact Customer Support"

    def undeploymodel(self, model_id):
        try:
            # 1- Copy model file
            str_model_id = str(model_id)
            path = os.path.join(deployment_folder, model_id)
            destination = "%s%s%s%s%s" % (deployment_folder, str_model_id, '/', str_model_id, '_model.pkl')
            os.remove(destination)
            os.rmdir(path)
            print("File deleted successfully.")

            # 2- Update deployment status
            model_profile = ModelProfile.query.filter_by(model_id=model_id).first()
            model_profile.deployed = 24
            db.session.commit()

            return "Model un-deployed successfully"
        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
            return "Model already deployed"

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
            return "There is no permission to deploy the model"

        # For other errors
        except Exception as e:
            print(e)
            print("Error occurred while copying file.")
            return "Failed to un-deploy the model, please try again or contact Customer Support"

    @staticmethod
    def get_dataset_info(dataFile):
        try:
            """
            Provide the columns data information as DataFrame object {'column':'column name', 'attributes':'column's attribute'}
            @rtype: DataFrame
            """
            data = pd.read_csv(dataFile)
            orginal_datacolumns = data.columns
            report = {}
            logging.info("------Numric columns")
            if len(data.select_dtypes(include=np.int).dtypes.values) > 0 or len(
                    data.select_dtypes(include=np.float).dtypes.values) > 0:
                numric_cols = pd.DataFrame(data.describe())
                numric_cols_names = numpy.array(numric_cols.columns)
                for col in numric_cols_names:
                    bb = ""
                    for i, row in numric_cols.iterrows():
                        bb = "{0}{1}: {2},\n ".format(bb, i, str(round(row[col], 2)))
                    report[col] = bb

            logging.info("------Object columns")
            if len(data.select_dtypes(include=np.object).dtypes.values) > 0:
                object_cols = pd.DataFrame(data.describe(include=object))
                object_cols_names = numpy.array(object_cols.columns)
                for col in object_cols_names:
                    bb = ""
                    for i, row in object_cols.iterrows():
                        bb = "{0}{1}: {2},\n ".format(bb, i, str(row[col]))
                    report[col] = bb

            report_df = pd.DataFrame(list(report.items()))
            report_df.rename(columns={0: 'column', 1: 'attributes'}, inplace=True)
            new_report_df = pd.DataFrame(columns=report_df['column'])
            r_row = numpy.array(pd.Series(report_df['attributes']))
            new_report_df.loc[0] = r_row
            new_report_df = new_report_df.reindex(numpy.array(orginal_datacolumns), axis=1)

            return new_report_df

        except Exception as e:
            logging.error(e)
            abort(500)
