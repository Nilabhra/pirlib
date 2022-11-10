import argparse
import base64
import os
import pickle
import sys
from tkinter import W
from typing import Any, Dict, Optional

import yaml

import pirlib.pir
from pirlib.backends import Backend
from pirlib.handlers.v1 import HandlerV1Context, HandlerV1Event


def encode(x):
    return base64.b64encode(pickle.dumps(x)).decode()


def decode(x):
    return pickle.loads(base64.b64decode(x.encode()))


def create_nfs_volume_spec(
    name: str, path_env_var: str, readonly: bool = False
) -> Dict[str, Any]:
    """Creates Kubernetes specs for defining NFS volumes.

    :param name: Name of the volume, required for mount reference.
    :type name: str
    :param path_env_var: Enviroment variable containing the NFS path of the
    directory/file.
    :type path_env_var: str
    :param readonly: Defines if the volume should be read only, defaults to False
    :type readonly: bool, optional
    :return: K8s specification for the NFS volume.
    :rtype: Dict[str, Any]
    """
    spec = {"name": name}
    nfs = {
        "server": os.environ("NFS_SERVER"),
        "path": os.environ(path_env_var),
        "readOnly": "yes" if readonly else "no",
    }
    spec["nfs"] = nfs
    return spec


def create_template_from_node(
    graph_inputs_encoded: str, node: pirlib.pir.Node
) -> Dict[str, Any]:
    """Generates an Argo template dictionary from the provided Node.

    :param graph_inputs_encoded: base64 encoding of the graph inputs.
    :type graph_inputs_encoded: str
    :param node: `Node` of a graph.
    :type node: pirlib.pir.Node
    :return: A dictionary containing the fields required to generate
    an Argo template for the given node.
    :rtype: Dict[str, Any]
    """
    name = node.id
    image = node.entrypoints["main"].image
    command = [
        "python",
        "-m",
        __name__,
        "node",
        encode(node),
        encode(graph_inputs_encoded),
    ]

    # Define the volume to be mounted.
    host_output_dir = os.environ("OUTPUT")
    nfs_server = os.environ("NFS_SERVER")
    # volumes = [{"name": "node_outputs", ""}]
    volume_mounts = [{"name": "node_outputs", "mountPath": "/mnt/node_outputs"}]

    # If the node has a valid graph input source,
    # mount the respective host system's file to the input volume

    # TODO: Read env vars and map the values to the input volume

    # Create the template dictionary.
    template = {
        "name": name,
        "container": {
            "image": image,
            "command": command,
            "volumeMounts": [volume],
        },
    }

    print(yaml.dump(template, default_flow_style=False, sort_keys=False))
    return template


def create_template_from_graph(
    graph_outputs_encoded: str, graph: pirlib.pir.Graph
) -> Dict[str, Any]:
    name = graph.id
    print(name)


class ArgoBatchBackend(Backend):
    def execute_parser(self) -> Optional[argparse.ArgumentParser]:
        pass

    def execute(self) -> Optional[argparse.ArgumentParser]:
        pass

    def generate_parser(self) -> Optional[argparse.ArgumentParser]:
        pass

    def generate(
        self,
        package: pirlib.pir.Package,
        config: Optional[dict] = None,
        args: Optional[argparse.Namespace] = None,
    ) -> Any:

        output_name = args.output.parts[-1].strip(".yml")

        workflow = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {"generateName": f"{output_name}"},
            "spec": {},
        }

        # Currently only 1 graph is supported
        assert len(package.graphs)
        graph = package.graphs[0]  # FIXME: need to handle multiple graphs
        print(graph)
        templates = []
        graph_inputs_encoded = encode(graph.inputs)
        graph_outputs_encoded = encode(graph.outputs)
        for i, node in enumerate(graph.nodes):
            # Using the first node as the entrypoint of the workflow.
            if i == 0:
                entrypoint = node.id

            # Creating a template for the current node.
            template = create_template_from_node(graph_inputs_encoded, node)
