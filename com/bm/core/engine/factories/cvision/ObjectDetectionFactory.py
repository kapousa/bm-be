import numpy
from flask import session, render_template

from app.modules.base.app_routes.directors.cvision.ObjectDetectionDirector import ObjectDetectionDirector


class ObjectDetectionFactory:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def createomodel(self, request):
        session['ds_soure'] = request.form.get('ds_source')
        objectdetectiondirector = ObjectDetectionDirector()

        return objectdetectiondirector.createobjectdetection(session['ds_goal'], session['ds_soure'])

    def showmodeldashboard(self, profile):
        objectdetectiondirector = ObjectDetectionDirector()

        return objectdetectiondirector.showobjdetectrmodeldashboard(profile)

    def detectobjects(self, model_id, request):
        opt_param = len(request.form)

        if opt_param == 0:
            # response = make_response()
            return render_template('applications/pages/cvision/objectdetection/labelfiles.html',
                                   message='No',model_id=model_id,
                                   download="#")

        host = request.form.get("ftp_host")
        uname = request.form.get("ftp_username")
        pword = request.form.get("ftp_password")
        webcam = 32 if (len(numpy.array(request.form.getlist("webcam"))) != 0) else 31
        runid = request.form.get("run_id")
        desc = request.form.get("desc")
        objectdetectiondirector = ObjectDetectionDirector()

        return objectdetectiondirector.detect_object(model_id, runid, desc, host, uname, pword, webcam)