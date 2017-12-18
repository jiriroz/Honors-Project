import datetime

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404
from django.template import loader
from django.urls import reverse

from .models import Question
import delaysapp.engine.modelapi as modelapi

import pickle
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AIRPORTDESC = pickle.load(open(os.path.join(BASE_DIR, "engine/keys/airportDesc.p"), "rb"))

def queryDelays(request):
    template = loader.get_template("delaysapp/input.html")
    return HttpResponse(template.render(dict(), request))

def computeDelay(request):
    try:
        flNum = request.POST["flNum"]
        origin = request.POST["airport"]
    except KeyError:
        return HttpResponse("You didn't select a choice")
    data, error = modelapi.getDelayForFlight(flNum, origin, datetime.date(2017, 12, 12))
    delay, dest = data
    context = {
        "flNum": flNum,
        "origin": origin,
        "destination": dest,
        "originlong":AIRPORTDESC[origin].split(":")[0],
        "destlong":AIRPORTDESC[dest].split(":")[0],
        "delay": round(delay, 2),
        "error": error
    }
    template = loader.get_template("delaysapp/output.html")
    return HttpResponse(template.render(context, request))

def index(request):
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    template = loader.get_template("delaysapp/index.html")
    context = {
        "latest_question_list": latest_question_list,
    }
    return HttpResponse(template.render(context, request))

def detail(request, question_id):
    try:
        question = Question.objects.get(pk=question_id)
    except Question.DoesNotExist:
        raise Http404("Question does not exist")
    return render(request, 'delaysapp/detail.html', {'question': question})

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'delaysapp/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('results', args=(question.id,)))

def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'delaysapp/results.html', {'question': question})


