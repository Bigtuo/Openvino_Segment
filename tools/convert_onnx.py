import torch
import onnxsim

def convert_to_onnx(self, simplify, model_path):
    import onnx
    self.generate(onnx=True)

    im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
    input_layer_names   = ["images"]
    output_layer_names  = ["output"]

    # Export the model
    print(f'Starting export with onnx {onnx.__version__}.')
    torch.onnx.export(self.net,
                    im,
                    f               = model_path,
                    verbose         = False,
                    opset_version   = 12,
                    training        = torch.onnx.TrainingMode.EVAL,
                    do_constant_folding = True,
                    input_names     = input_layer_names,
                    output_names    = output_layer_names,
                    dynamic_axes    = None)

    # Checks
    model_onnx = onnx.load(model_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify onnx
    if simplify:
        import onnxsim
        print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, model_path)

    print('Onnx model save as {}'.format(model_path))