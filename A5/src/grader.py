#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import os
import traceback
import torch
import torch.nn as nn
import timeout_decorator
# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

# ----------
# CONSTANTS
# ----------
BLOCK_SIZE = 128
PRETRAIN_CORPUS_PATH = './data/wiki.txt'
PRETRAIN_TEXT = open(PRETRAIN_CORPUS_PATH, encoding='utf-8').read()

def score_preds(preds_fn, answers_fn):
    n_correct, n_total = 0, 0
    with open(preds_fn, "r", encoding='utf-8') as f_stu:
        with open(answers_fn, "r", encoding='utf-8') as f_sol:
            for sol_l, stu_l in zip(f_sol, f_stu):
                _, a_sol = sol_l.strip().split("\t")
                a_stu = stu_l.strip()
                n_correct += (1 if a_sol == a_stu else 0)
                n_total += 1
    return n_correct, n_total
  
class sample_GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    synthesizer = False

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

#########
# TESTS #
#########
class Test_1c(GradedTestCase):
  def setUp(self):
    self.pretrain_dataset = submission.CharCorruptionDataset(PRETRAIN_TEXT, BLOCK_SIZE)
    self.mconf = submission.GPTConfig(self.pretrain_dataset.vocab_size, self.pretrain_dataset.block_size, n_layer=4, n_head=8, n_embd=256)
    self.vanilla_model = submission.initialize_vanilla_model(self.mconf)

  @graded(is_hidden=True)
  def test_0(self):
    """1c-0-hidden:  vanilla model similarity"""
    expected = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).initialize_vanilla_model(self.mconf)
    self.assertEqual(str(expected), str(self.vanilla_model))
  
  @graded()
  @timeout_decorator.timeout(15)
  def test_1(self):
    """1c-1-basic:  correct trainer object initialization for finetune without pretraining"""
    student_trainer_conf, student_trainer = submission.finetune(None, './data/birth_places_train.tsv', self.pretrain_dataset, BLOCK_SIZE, self.vanilla_model)

    self.assertEqual(student_trainer_conf.max_epochs, 75)
    self.assertEqual(student_trainer_conf.batch_size, 256)
    self.assertEqual(student_trainer_conf.learning_rate, 0.0006)
    self.assertEqual(student_trainer_conf.betas, (0.9, 0.95))
    self.assertEqual(student_trainer_conf.grad_norm_clip, 1.0)
    self.assertEqual(student_trainer_conf.weight_decay, 0.1)
    self.assertEqual(student_trainer_conf.lr_decay,True)
    self.assertEqual(student_trainer_conf.warmup_tokens, 10240)
    self.assertEqual(student_trainer_conf.final_tokens, 75212800)
    self.assertEqual(student_trainer_conf.ckpt_path, None)
    self.assertEqual(student_trainer_conf.num_workers, 4)

class Test_1d(GradedTestCase):
  @graded(is_hidden=True)
  def test_0(self):
    """1d-0-hidden:   test the dev score for vanilla attention without pretrain"""
    n_correct, n_total = score_preds(
        "./submission/vanilla.nopretrain.dev.predictions",
        "./data/birth_dev.tsv")
    self.assertGreaterEqual(n_correct, 1)
  
  @graded(is_hidden=True)
  def test_1(self):
    #TODO: Make sure to place this under the autograder code section birth_test.tsv not exposed to students
    """1d-1-hidden:   test the test score for vanilla attention without pretrain"""
    n_correct, n_total = score_preds(
        "./submission/vanilla.nopretrain.test.predictions",
        "./data/birth_test.tsv")
    self.assertGreaterEqual(n_correct, 1)

class Test_1e(GradedTestCase):
  def setUp(self):
    self.data = ("let me take you down\ncause I'm going to\n"
       "strawberry fields\nnothing is real and nothing to get hung about")
    for i in range(100):
        self.data += "\n" + "a"*random.randint(7, 42)
    self.dataset = submission.CharCorruptionDataset(self.data, 50)

  @graded()
  def test_0(self):
    """1e-0-basic:  check CharCorruptionDataset truncation length"""
    invalid_len = False

    len_fracs = []

    for (x, y), entry in zip(self.dataset, self.data.split("\n")):
        x_unpad = [c.item() for c in x if c >= 2] # remove padding and mask chars
        l = len(x_unpad)
        if l < 4 or l > int(50*7/8):
            invalid_len = True

        len_fracs.append(l / len(entry))

    self.assertEqual(invalid_len, False)
    self.assertGreater(np.std(len_fracs), 0.05)

  @graded()
  def test_1(self):
    """1e-1-basic:  check CharCorruptionDataset rearrange"""
    format_ok = True

    for (x, y), entry in zip(self.dataset, self.data.split("\n")):
        x = [c.item() for c in x]
        # padding should only be at end
        seen_pad = False
        for i, e in enumerate(x):
            if e == 0:
                if not seen_pad:
                    body = x[:i]
                seen_pad = True
            else:
                if seen_pad:
                    format_ok = False
                    break
        body_s = "".join([self.dataset.itos[e] for e in body])
        toks = body_s.split(self.dataset.MASK_CHAR)
        if len(toks) not in [3, 4]:    # it's ok for mask char to be after masked content
            format_ok = False
            break

        if len(toks) == 4 and toks[3] != "":
            format_ok = False
            break

        orig = toks[0] + toks[2] + toks[1]
        if orig != entry[:len(orig)]:
            format_ok = False
            break

    self.assertEqual(format_ok, True)
  
  @graded()
  def test_2(self):
    """1e-2-basic:  check CharCorruptionDataset io"""
    is_ok = True

    for (x, y), entry in zip(self.dataset, self.data.split("\n")):
        for xe, ye in zip(x[1:], y[:-1]):
            if not xe.item() == ye.item():
                is_ok = False

    self.assertEqual(is_ok, True)
  
  @graded()
  def test_3(self):
    """1e-3-basic:  check CharCorruptionDataset masked content length"""
    lens = []
    true_lens = []

    for (x, y), entry in zip(self.dataset, self.data.split("\n")):
        x = [c.item() for c in x if c.item() != 0]
        body_s = "".join([self.dataset.itos[e] for e in x])
        toks = body_s.split(self.dataset.MASK_CHAR)

        lens.append(len(toks[2]))
        true_lens.append((len(body_s) - 2) * 0.25)

    self.assertLessEqual(np.abs(np.mean(np.array(lens) - np.array(true_lens))), 1.5)
    self.assertGreater(np.std(lens), 0.01)
    

class Test_1f(GradedTestCase):
  def setUp(self):
    self.pretrain_dataset = submission.CharCorruptionDataset(PRETRAIN_TEXT, BLOCK_SIZE)
    self.mconf = submission.GPTConfig(self.pretrain_dataset.vocab_size, self.pretrain_dataset.block_size, n_layer=4, n_head=8, n_embd=256)
    self.vanilla_model = submission.initialize_vanilla_model(self.mconf)

  @graded()
  def test_0(self):
    """1f-0-basic:  check basic vanilla pretrain trainer object"""
    student_trainer_conf, student_trainer = submission.pretrain(self.pretrain_dataset, BLOCK_SIZE, self.vanilla_model)
    
    self.assertEqual(student_trainer_conf.max_epochs, 650)
    self.assertEqual(student_trainer_conf.batch_size, 128)
    self.assertEqual(student_trainer_conf.learning_rate, 6e-3)
    self.assertEqual(student_trainer_conf.betas, (0.9, 0.95))
    self.assertEqual(student_trainer_conf.grad_norm_clip, 1.0)
    self.assertEqual(student_trainer_conf.weight_decay, 0.1)
    self.assertEqual(student_trainer_conf.lr_decay,True)
    self.assertEqual(student_trainer_conf.warmup_tokens, 10240)
    self.assertEqual(student_trainer_conf.final_tokens, 75212800)
    self.assertEqual(student_trainer_conf.ckpt_path, None)
    self.assertEqual(student_trainer_conf.num_workers, 4)
  
  
  @graded(is_hidden=True)
  def test_2(self):
    """1f-0-hidden:   test the dev score for vanilla attention with pretrain"""
    n_correct, n_total = score_preds(
        "./submission/vanilla.pretrain.dev.predictions",
        "./data/birth_dev.tsv")
    self.assertGreaterEqual(n_correct / n_total, 0.1)

  @graded(is_hidden=True)
  #TODO: Change the path of birth_test.tsv for GradeScope autograder
  def test_3(self):
    """1f-1-hidden:   test the test score for vanilla attention with pretrain"""
    n_correct, n_total = score_preds(
        "./submission/vanilla.pretrain.test.predictions",
        "./data/birth_test.tsv")
    self.assertGreaterEqual(n_correct / n_total, 0.09)
  
class Test_1g(GradedTestCase):
  def setUp(self):
    self.pretrain_dataset = submission.CharCorruptionDataset(PRETRAIN_TEXT, BLOCK_SIZE)
    self.mconf = submission.GPTConfig(self.pretrain_dataset.vocab_size, self.pretrain_dataset.block_size, n_layer=4, n_head=8, n_embd=256)
    self.synthesizer_model = submission.initialize_synthesizer_model(self.mconf)

  @timeout_decorator.timeout(15)
  @graded()
  def test_0(self):
    """1g-0-basic:  correct trainer object initialization for finetune with pretraining for synthesizer"""
    student_trainer_conf, student_trainer = submission.finetune('./submission/synthesizer.pretrain.params', './data/birth_places_train.tsv', self.pretrain_dataset, BLOCK_SIZE, self.synthesizer_model)
    
    self.assertEqual(student_trainer_conf.max_epochs, 10)
    self.assertEqual(student_trainer_conf.batch_size, 256)
    self.assertEqual(student_trainer_conf.learning_rate, 0.0006)
    self.assertEqual(student_trainer_conf.betas, (0.9, 0.95))
    self.assertEqual(student_trainer_conf.grad_norm_clip, 1.0)
    self.assertEqual(student_trainer_conf.weight_decay, 0.1)
    self.assertEqual(student_trainer_conf.lr_decay,True)
    self.assertEqual(student_trainer_conf.warmup_tokens, 10240)
    self.assertEqual(student_trainer_conf.final_tokens, 75212800)
    self.assertEqual(student_trainer_conf.ckpt_path, None)
    self.assertEqual(student_trainer_conf.num_workers, 4)

  @graded(is_hidden=True)
  def test_1(self):
    """1g-0-hidden:   test the dev score for synthesizer attention with pretrain"""
    n_correct, n_total = score_preds(
        "./submission/synthesizer.pretrain.dev.predictions",
        "./data/birth_dev.tsv")
    self.assertGreaterEqual(n_correct / n_total, 0.05)
  
  @graded(is_hidden=True)
  def test_2(self):
    """1g-1-hidden:   test the test score for synthesizer attention with pretrain"""
    n_correct, n_total = score_preds(
        "./submission/synthesizer.pretrain.test.predictions",
        "./data/birth_test.tsv")
    self.assertGreaterEqual(n_correct / n_total, 0.04)
  
  @graded(is_hidden=True)
  def test_3(self):
    """1g-2-hidden:   check if synthesizer attention values match"""
    mconf = sample_GPTConfig(5, 8, n_layer=1, n_head=3, n_embd=6)
    att_student = submission.SynthesizerAttention(mconf)
    att_expected = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).SynthesizerAttention(mconf)
    att_student.eval()
    att_expected.eval()
    att_student.load_state_dict(att_expected.state_dict())
    with torch.no_grad():
        x = torch.randn(11, 7, 6)
        y_sol = att_expected(x)
        y_stu = att_student(x)
    self.assertLess(torch.norm(y_sol - y_stu), 1e-8)

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)