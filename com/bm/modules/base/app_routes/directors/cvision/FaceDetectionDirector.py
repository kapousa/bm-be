from flask import render_template, request

from app.modules.base.constants.BM_CONSTANTS import docs_templates_folder, output_docs
from app.modules.base.db_models.ModelCvisionRun import ModelCvisionRun
from app.modules.base.db_models.ModelProfile import ModelProfile
from app.modules.base.db_models.ModelAPIDetails import ModelAPIDetails
from com.bm.apis.v1.APIHelper import APIHelper
from com.bm.controllers.cvision.FaceDetectionCotroller import FaceDetectionCotroller
from com.bm.controllers.cvision.ObjectDetectionCotroller import ObjectDetectionCotroller


class FaceDetectionDirector:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def createfacedetection(self, ds_goal, ds_source):
        try:
            facedetectingcotroller = FaceDetectionCotroller()
            facedetectionmodel = facedetectingcotroller.create_model(ds_goal, ds_source)

            page_url =  "{0}cvision/{1}/facedetect/detect".format(request.host_url, str(facedetectionmodel['model_id']))
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

            # APIs details and create APIs document
            model_api_details = ModelAPIDetails.query.first()
            apihelper = APIHelper()
            model_head = ModelProfile.query.with_entities(ModelProfile.model_id, ModelProfile.model_name).filter_by(
                model_id=facedetectionmodel['model_id']).first()
            generate_apis_docs = apihelper.generateapisdocs(model_head.model_id,
                                                            str(request.host_url + 'api/' + model_api_details.api_version),
                                                            docs_templates_folder, output_docs)

            return render_template('applications/pages/cvision/facedetection/modelstatus.html',
                                   message='There is no active model',
                                   fname=facedetectionmodel['model_name'],page_url=page_url, page_embed=page_embed,
                                   segment='createmodel', model_id=facedetectionmodel['model_id'],
                                   created_on=facedetectionmodel['created_on'],
                                   updated_on=facedetectionmodel['updated_on'],
                                   last_run_time=facedetectionmodel['last_run_time'],
                                   ds_goal=ds_goal, ds_sourc=ds_source
                                   )
        except Exception as e:
            print(e)
            return 0

    def showobjdetectrmodeldashboard(self,profile):
        page_url = "{0}cvision/{1}/objtdetect/detect".format(request.host_url, str(profile['model_id']))
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
        run_history = ModelCvisionRun.query.filter_by( model_id = str(profile['model_id'])).all()
        return render_template('applications/pages/cvision/objectdetection/dashboard.html',
                               message='No', run_history=run_history,
                               fname=profile['model_name'],page_url=page_url, page_embed=page_embed,
                               segment='showdashboard', created_on=profile['created_on'],
                               ds_goal=profile['ds_goal'],model_id=profile['model_id'],
                               updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])

    def detect_object(self, model_id, runid, desc, host, uname, pword):
        try:
            objectdetectioncontroller = ObjectDetectionCotroller()
            run_identifier = "%s%s%s" % (model_id, '_', runid)
            labelfileslink, labeled = objectdetectioncontroller.labelfiles(run_identifier, desc, host, uname, pword, 27)

            return render_template('applications/pages/cvision/objectdetection/labelfiles.html',
                                   message='No', labeled = labeled, model_id=model_id, run_id=runid,
                                   downloadlink= labelfileslink)
        except Exception as e:
            print(e)
            return 0



