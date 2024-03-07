import xml.etree.ElementTree as ET
import finn.util.basic as fbasic
import finn.transformation.fpgadataflow.templates as ftemplates


def is_board_exists(board: str):
    return board in fbasic.pynq_part_map.keys()


def add_board(
    board_xml_path: str,
    port_width: int = 32,
    template_board: str = None,
) -> str:
    """
    Add a FINN board definition.
    This function does not persist changes, so it must be called every time the interpreter is relaunched.

    Params:
        - board_xml_path : Path to the board.xml of the new board, eg `f'{os.environ["FINN_ROOT"]}/deps/board_files/zybo-z7-20/A.0/board.xml'`
        - port_width : Check `finn.util.basic.pynq_native_port_width`: native AXI HP port width (in bits) for PYNQ boards
        - template_board : If set, `port_width` from the template board

    Returns the name of the new board, parsed from the XML
    """

    # Parse board definition XML
    tree = ET.parse(board_xml_path)
    root = tree.getroot()

    # Find some needed informations from board
    root_attr = root.attrib  # root node atts
    xml_board_name = root_attr["name"]
    xml_board_ver = root.find("file_version").text
    xml_part_name = ""  # eg xc7z020....
    xml_name = ""  # part0 in most cases
    for component in root.find("components").iter("component"):
        if "type" not in component.attrib.keys():
            continue
        if component.attrib["type"] != "fpga":
            continue
        xml_part_name = component.attrib["part_name"]
        xml_name = component.attrib["name"]
        break

    if xml_board_name in fbasic.pynq_part_map.keys():
        raise ValueError(
            f"Board {xml_board_name} already exists in `finn.util.basic.pynq_part_map`"
        )

    # Patch board definitions
    if template_board is not None:
        if fbasic.pynq_part_map[template_board] != xml_part_name:
            raise ValueError(
                f"PYNQ template PN ({fbasic.pynq_part_map[template_board]}) does not match XML PN ({xml_part_name})"
            )
        fbasic.pynq_native_port_width[xml_board_name] = fbasic.pynq_native_port_width[
            template_board
        ]
        fbasic.pynq_part_map[xml_board_name] = fbasic.pynq_part_map[template_board]
    else:
        fbasic.pynq_native_port_width[xml_board_name] = port_width
        fbasic.pynq_part_map[xml_board_name] = xml_part_name

    # Patch ip config script template
    zynq_type = (
        "zynq_7000"
        if fbasic.pynq_part_map[xml_board_name].startswith("xc7z")
        else (
            "zynq_us+"
            if fbasic.pynq_part_map[xml_board_name].startswith("xc")
            else None
        )
    )

    if zynq_type is None:
        raise ValueError(
            f"Cannot add a board that is not Zynq 7000 or UltraScale+ based ({fbasic.pynq_part_map[xml_board_name]})"
        )

    new_board_tcl = ['} elseif {$BOARD == "%s"} {' % xml_board_name]
    new_board_tcl.append('    set ZYNQ_TYPE "%s"' % zynq_type)
    new_board_tcl.append(
        "    set_property board_part %s:%s:%s:%s [current_project]"
        % (
            root_attr["vendor"],
            xml_board_name,
            xml_name,
            xml_board_ver,
        )
    )
    # print("\n".join(new_board_tcl))

    tmp = ftemplates.custom_zynq_shell_template.splitlines()
    idx = tmp.index('} elseif {$BOARD == "Pynq-Z1"} {')
    for line in new_board_tcl:
        tmp.insert(idx, line)
        idx = idx + 1
    # print("\n".join(tmp[idx - 4 : idx + 3]))
    ftemplates.custom_zynq_shell_template = "\n".join(tmp)

    return xml_board_name
