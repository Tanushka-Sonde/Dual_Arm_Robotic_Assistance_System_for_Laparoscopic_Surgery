"""
generate_tdcr_xml.py
====================
Generates a MuJoCo XML for the two-Panda + TDCR system with a user-supplied
TDCR scale factor.

Usage
-----
    python generate_tdcr_xml.py                   # interactive prompt
    python generate_tdcr_xml.py 0.6               # positional arg
    python generate_tdcr_xml.py --scale 1.0       # flag
    python generate_tdcr_xml.py 0.8 --out my.xml  # custom output file

The Panda arm geometry is NEVER rescaled — only the TDCR components are
affected.

Original TDCR reference values (scale = 1.0)
--------------------------------------------
  platform_mesh / base_mesh / segment_mesh  : 0.001
  segment_2_mesh  : 0.00091266
  segment_3_mesh  : 0.00083295
  segment_4_mesh  : 0.00076019
  segment_5_mesh  : 0.00069380
  segment_6_mesh  : 0.00063320
  segment_7_mesh  : 0.00057790
  segment_8_mesh  : 0.00052742
  segment_9_mesh  : 0.00048135
  segment_10_mesh : 0.00043931
  segment_11_mesh : 0.00040094
  segment_12_mesh : 0.00036592
  gripper meshes  : 1.0  (body pos only, mesh scale left at 1)

  Geometric positions and joint offsets are also multiplied by SCALE.
  Tendon / muscle lengthrange:
    arm     : 0.5–2.0      × SCALE
    gripper : 0.009–0.0166 × SCALE
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# Original (unscaled) TDCR geometry constants
# ---------------------------------------------------------------------------

MESH_SCALES_ORIG = {
    "platform_mesh":   0.001,
    "base_mesh":       0.001,
    "segment_mesh":    0.001,
    "segment_2_mesh":  0.00091266,
    "segment_3_mesh":  0.00083295,
    "segment_4_mesh":  0.00076019,
    "segment_5_mesh":  0.00069380,
    "segment_6_mesh":  0.00063320,
    "segment_7_mesh":  0.00057790,
    "segment_8_mesh":  0.00052742,
    "segment_9_mesh":  0.00048135,
    "segment_10_mesh": 0.00043931,
    "segment_11_mesh": 0.00040094,
    "segment_12_mesh": 0.00036592,
}

# Body positions along the TDCR chain (z offsets, original values)
BODY_POS_Z_ORIG = {
    "tdcr_base":   (0, 0.070,  0.140),   # (x, y, z)
    "segment_1":   (0, 0.126,  0.2362),
    "segment_2":   (0, 0,     -0.0782),
    "segment_3":   (0, 0,     -0.07135),
    "segment_4":   (0, 0,     -0.06511),
    "segment_5":   (0, 0,     -0.0594),
    "segment_6":   (0, 0,     -0.05420),
    "segment_7":   (0, 0,     -0.0495),
    "segment_8":   (0, 0,     -0.04515),
    "segment_9":   (0, 0,     -0.0412),
    "segment_10":  (0, 0,     -0.0376),
    "segment_11":  (0, 0,     -0.0343),
    "segment_12":  (0, 0,     -0.0313),
}

JOINT_POS_Z_ORIG = {
    "Joint1":  0.086,
    "Joint2":  0.0782,
    "Joint3":  0.07138,
    "Joint4":  0.06537634,
    "Joint5":  0.0596668,
    "Joint6":  0.0544552,
    "Joint7":  0.0496994,
    "Joint8":  0.04535812,
    "Joint9":  0.0413961,
    "Joint10": 0.03778066,
    "Joint11": 0.03448084,
    "Joint12": 0.03146912,
}

# Stiffness / damping — NOT scaled (physical material property)
JOINT_PARAMS = {
    "Joint1":  dict(stiffness=0.081647,  damping=0.083295),
    "Joint2":  dict(stiffness=0.034692,  damping=0.00693847),
    "Joint3":  dict(stiffness=0.0288984, damping=0.00577975),
    "Joint4":  dict(stiffness=0.0240724, damping=0.00481453),
    "Joint5":  dict(stiffness=0.0200523, damping=0.0040105),
    "Joint6":  dict(stiffness=0.0167035, damping=0.00334075),
    "Joint7":  dict(stiffness=0.0139141, damping=0.00278284),
    "Joint8":  dict(stiffness=0.0115904, damping=0.00231811),
    "Joint9":  dict(stiffness=0.00965481,damping=0.00193099),
    "Joint10": dict(stiffness=0.00804246, damping=0.00160851),
    "Joint11": dict(stiffness=0.00669937, damping=0.00133989),
    "Joint12": dict(stiffness=0.00558057, damping=0.00111613),
}

# Inertial params — mass not scaled, diaginertia scaled by SCALE^5 would be
# physically correct, but original XML keeps them fixed; we preserve that.
INERTIAL = {
    "segment_1":  dict(mass=0.007602,    dia="0.000364 0.000364 0.00045711"),
    "segment_2":  dict(mass=0.00577752,  dia="0.000303212 0.000303212 0.000380773"),
    "segment_3":  dict(mass=0.00439092,  dia="0.000252576 0.000252576 0.000317184"),
    "segment_4":  dict(mass=0.0033371,   dia="0.000210395 0.000210395 0.000264214"),
    "segment_5":  dict(mass=0.00253619,  dia="0.000175259 0.000175259 0.00022009"),
    "segment_6":  dict(mass=0.00192751,  dia="0.000145991 0.000145991 0.000183335"),
    "segment_7":  dict(mass=0.0014649,   dia="0.000121611 0.000121611 0.000152718"),
    "segment_8":  dict(mass=0.00111333,  dia="0.000101302 0.000101302 0.000127214"),
    "segment_9":  dict(mass=0.000846129, dia="8.43843e-05 8.43843e-05 0.000105969"),
    "segment_10": dict(mass=0.000643058, dia="7.02921e-05 7.02921e-05 8.82726e-05"),
    "segment_11": dict(mass=0.000488724, dia="5.85533e-05 5.85533e-05 7.3531e-05"),
    "segment_12": dict(mass=0.10037143,  dia="4.87749e-05 4.87749e-05 6.12514e-05"),
}

# Original site positions (x, z) for each segment — [+x/0.z1, -x/0.z1, +x2/0.z2, -x2/0.z2]
SITE_ORIG = {
    "segment_1":  [(0.069, 0.0525), (0.065, 0.0050)],
    "segment_2":  [(0.0635, 0.0500), (0.0600, 0.0050)],
    "segment_3":  [(0.058, 0.0450), (0.055, 0.0050)],
    "segment_4":  [(0.0525, 0.0400), (0.0500, 0.0050)],
    "segment_5":  [(0.0475, 0.0375), (0.0450, 0.0050)],
    "segment_6":  [(0.0425, 0.0350), (0.0400, 0.0050)],
    "segment_7":  [(0.0390, 0.0310), (0.0365, 0.0033)],
    "segment_8":  [(0.0350, 0.0295), (0.0327, 0.0026)],
    "segment_9":  [(0.0322, 0.0260), (0.0300, 0.0019)],
    "segment_10": [(0.0294, 0.0235), (0.0276, 0.00212)],
    "segment_11": [(0.0280, 0.0215), (0.0249, 0.0020)],
    "segment_12": [(0.0240, 0.0197), (0.0225, 0.0020)],
}

# Base site
BASE_SITE_X_ORIG = 0.0725
BASE_SITE_Z_ORIG = 0.020

# Gripper geometry originals
GRIPPER_POS_X_ORIG = 0.0231   # ±x of gripper_1/2
GRIPPER_POS_Z_ORIG = 0.0355   # z of gripper_1/2
GRIP_SITE_X_ORIG   = 0.0185   # grip_1/2 site -x
GRIPPER_BASE_SITE_Z_ORIG = 0.020  # base_1 site z

# Tendon length ranges (original)
ARM_TENDON_RANGE_ORIG    = (0.5, 2.0)
GRIPPER_TENDON_RANGE_ORIG = (0.009, 0.0166)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def f(v, digits=8):
    """Format a float compactly."""
    s = f"{v:.{digits}g}"
    return s

def ms(orig, scale):
    v = orig * scale
    return f"{v} {v} {v}"

def p(x, y, z):
    return f"{f(x)} {f(y)} {f(z)}"

def ind(level):
    return "  " * level

# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def build_xml(scale: float) -> str:
    S = scale

    # Scaled mesh scales
    def smsh(key):
        v = MESH_SCALES_ORIG[key] * S
        return f"{v} {v} {v}"

    # Scaled body positions
    def sbp(key):
        x, y, z = BODY_POS_Z_ORIG[key]
        return p(x * S, y * S, z * S)

    # Scaled joint z
    def sjz(key):
        return f(JOINT_POS_Z_ORIG[key] * S)

    # Scaled site positions
    def ss(seg, pair_idx, sign_x):
        (sx, sz), _ = SITE_ORIG[seg][0], SITE_ORIG[seg][1]
        bx, bz = SITE_ORIG[seg][pair_idx]
        return f"{f(sign_x * bx * S)} 0 {f(bz * S)}"

    def ss_pair(seg, pair_idx):
        bx, bz = SITE_ORIG[seg][pair_idx]
        return (f(bx * S), f(bz * S))

    # Tendon ranges
    arm_lo  = f(ARM_TENDON_RANGE_ORIG[0]    * S)
    arm_hi  = f(ARM_TENDON_RANGE_ORIG[1]    * S)
    grp_lo  = f(GRIPPER_TENDON_RANGE_ORIG[0] * S)
    grp_hi  = f(GRIPPER_TENDON_RANGE_ORIG[1] * S)

    arm_range = f"{arm_lo} {arm_hi}"
    grp_range = f"{grp_lo} {grp_hi}"

    # Base site
    bsx = f(BASE_SITE_X_ORIG * S)
    bsz = f(BASE_SITE_Z_ORIG * S)

    # Gripper
    gpx = f(GRIPPER_POS_X_ORIG * S)
    gpz = f(GRIPPER_POS_Z_ORIG * S)
    gsx = f(GRIP_SITE_X_ORIG   * S)
    gbsz = f(GRIPPER_BASE_SITE_Z_ORIG * S)

    # Gripper mesh scale (original = 1.0, scaled by S)
    gms = f"{S} {S} {S}"

    # -----------------------------------------------------------------------
    # Build TDCR body chain for one robot
    # prefix  : "" (p1) or "p2_"
    # jprefix : "tdcr_" or "p2_tdcr_"  (for platform/base body names)
    # -----------------------------------------------------------------------
    def tdcr_chain(prefix, jprefix):
        """Return the full TDCR XML string indented for inside link7."""
        bp = sbp("tdcr_base")
        sp1 = sbp("segment_1")

        lines = []
        L = 22  # base indent level (spaces)
        i0 = " " * L

        def li(extra, text):
            return " " * (L + extra * 2) + text

        def site4(seg, sname_tmpl, indent_extra):
            px1, pz1 = ss_pair(seg, 0)
            px2, pz2 = ss_pair(seg, 1)
            n1 = f"{sname_tmpl}_1"
            n2 = f"{sname_tmpl}_2"
            n3 = f"{sname_tmpl}_3"
            n4 = f"{sname_tmpl}_4"
            ie = indent_extra
            return [
                li(ie, f'<site name="{n1}" pos=" {px1} 0 {pz1}" size="0.002 0.002 0.002"/>'),
                li(ie, f'<site name="{n2}" pos="-{px1} 0 {pz1}" size="0.002 0.002 0.002"/>'),
                li(ie, f'<site name="{n3}" pos=" {px2} 0 {pz2}" size="0.002 0.002 0.002"/>'),
                li(ie, f'<site name="{n4}" pos="-{px2} 0 {pz2}" size="0.002 0.002 0.002"/>'),
            ]

        # attachment body
        lines.append(li(0, f'<body name="{prefix}attachment" pos="0 0 0.107" quat="0.3826834 0 0 0.9238795">'))
        lines.append(li(1,   f'<site name="{prefix}attachment_site"/>'))
        lines.append(li(1,   f'<body name="{jprefix}platform" pos="0 0 0" euler="0 0 0">'))
        lines.append(li(2,     '<geom type="mesh" mesh="platform_mesh" rgba="0.6 0.6 0.6 1"/>'))

        # tdcr_base
        base_pos = f'0 {f(0.042 * S / 0.6 * S / S)} {f(0.084 * S / 0.6 * S / S)}'
        # Recompute directly from originals
        base_y = f(0.070 * S)
        base_z = f(0.140 * S)
        lines.append(li(2, f'<body name="{jprefix}base" pos="0 {base_y} {base_z}" euler="-0.524 0 1.57">'))
        lines.append(li(3,   '<geom type="mesh" mesh="base_mesh" rgba="0.3 0.3 0.8 1"/>'))
        lines.append(li(3,   f'<site name="{prefix}b_1" pos="-{bsx} 0 {bsz}" size="0.002 0.002 0.002"/>'))
        lines.append(li(3,   f'<site name="{prefix}b_2" pos=" {bsx} 0 {bsz}" size="0.002 0.002 0.002"/>'))
        lines.append(li(2, f'</body>'))

        # segment_1
        s1y = f(0.126 * S)
        s1z = f(0.2362 * S)
        lines.append(li(2, f'<body name="{prefix}segment_1" pos="0 {s1y} {s1z}" euler="-0.524 3.14159 1.5708">'))
        lines.append(li(3,   '<geom type="mesh" mesh="segment_mesh" rgba="0.9 0.3 0.3 1"/>'))
        jp = JOINT_PARAMS["Joint1"]
        lines.append(li(3,   f'<joint name="{prefix}tdcr_Joint1" pos="0 0 {sjz("Joint1")}" axis="0 -1 0" limited="true" range="-0.523 0.523" stiffness="{jp["stiffness"]}" damping="{jp["damping"]}"/>'))
        ini = INERTIAL["segment_1"]
        lines.append(li(3,   f'<inertial pos="0 0 0" quat="1 0 0 0" mass="{ini["mass"]}" diaginertia="{ini["dia"]}"/>'))
        lines += site4("segment_1", f"{prefix}s1", 3)

        segs = [
            ("segment_2",  "segment_2_mesh",  "segment_2",  "Joint2",  2),
            ("segment_3",  "segment_3_mesh",  "segment_3",  "Joint3",  3),
            ("segment_4",  "segment_4_mesh",  "segment_4",  "Joint4",  4),
            ("segment_5",  "segment_5_mesh",  "segment_5",  "Joint5",  5),
            ("segment_6",  "segment_6_mesh",  "segment_6",  "Joint6",  6),
            ("segment_7",  "segment_7_mesh",  "segment_7",  "Joint7",  7),
            ("segment_8",  "segment_8_mesh",  "segment_8",  "Joint8",  8),
            ("segment_9",  "segment_9_mesh",  "segment_9",  "Joint9",  9),
            ("segment_10", "segment_10_mesh", "segment_10", "Joint10", 10),
            ("segment_11", "segment_11_mesh", "segment_11", "Joint11", 11),
            ("segment_12", "segment_12_mesh", "segment_12", "Joint12", 12),
        ]

        depth = 3  # extra indent units from attachment body
        for bname, mname, skey, jname, snum in segs:
            bz = f(BODY_POS_Z_ORIG[bname][2] * S)
            jp = JOINT_PARAMS[jname]
            ini = INERTIAL[bname]
            depth += 1
            lines.append(li(depth, f'<body name="{prefix}{bname}" pos="0 0 {bz}">'))
            lines.append(li(depth+1, f'<geom type="mesh" mesh="{mname}" rgba="0.9 0.3 0.3 1"/>'))
            lines.append(li(depth+1, f'<joint name="{prefix}tdcr_{jname}" pos="0 0 {sjz(jname)}" axis="0 -1 0" limited="true" range="-0.523 0.523" stiffness="{jp["stiffness"]}" damping="{jp["damping"]}"/>'))
            lines.append(li(depth+1, f'<inertial pos="0 0 0" quat="1 0 0 0" mass="{ini["mass"]}" diaginertia="{ini["dia"]}"/>'))
            lines += site4(bname, f"{prefix}s{snum}", depth + 1)

        # Gripper base inside segment_12
        lines.append(li(depth+1, f'<body name="{prefix}tdcr_gripper_base" pos="0 0 0" euler="0 3.14159 0">'))
        lines.append(li(depth+2,   '<geom type="mesh" mesh="gripper_base" rgba="0.6 0.6 0.6 1"/>'))
        lines.append(li(depth+2,   f'<site name="{prefix}base_1" pos="0 0 {gbsz}" size="0.002 0.002 0.002"/>'))
        lines.append(li(depth+2,   f'<body name="{prefix}gripper_1" pos="{gpx} 0 {gpz}">'))
        lines.append(li(depth+3,     '<geom type="mesh" mesh="gripper" rgba="0.9 0.9 0.9 1" friction="2 0.01 0.001"/>'))
        lines.append(li(depth+3,     '<joint name="{0}gripper_1_joint" type="hinge" pos="0 0 0" axis="0 1 0" range="-45 0" limited="true" stiffness="5" damping="0.5"/>'.format(prefix)))
        lines.append(li(depth+3,     f'<site name="{prefix}grip_1" pos="-{gsx} 0 0" size="0.002 0.002 0.002"/>'))
        lines.append(li(depth+2,   '</body>'))
        lines.append(li(depth+2,   f'<body name="{prefix}gripper_2" pos="-{gpx} 0 {gpz}" euler="0 0 3.14159">'))
        lines.append(li(depth+3,     '<geom type="mesh" mesh="gripper" rgba="1 1 1 1" friction="2 0.01 0.001"/>'))
        lines.append(li(depth+3,     '<joint name="{0}gripper_2_joint" type="hinge" pos="0 0 0" axis="0 -1 0" range="0 45" limited="true" stiffness="5" damping="0.5"/>'.format(prefix)))
        lines.append(li(depth+3,     f'<site name="{prefix}grip_2" pos="-{gsx} 0 0" size="0.002 0.002 0.002"/>'))
        lines.append(li(depth+2,   '</body>'))
        lines.append(li(depth+1, '</body><!-- gripper_base -->'))

        # Close all nested segment bodies
        for bname, _, _, _, _ in reversed(segs):
            lines.append(li(depth, f'</body><!-- {prefix}{bname} -->'))
            depth -= 1

        # Close segment_1
        lines.append(li(3, f'</body><!-- {prefix}segment_1 -->'))
        lines.append(li(2, f'</body><!-- {jprefix}platform -->'))
        lines.append(li(1, f'</body><!-- {prefix}attachment -->'))

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Tendon XML helpers
    # -----------------------------------------------------------------------
    def arm_tendon(tname, color, b_site, sites_prefix, rgba):
        lines = [
            f'    <spatial name="{tname}" width="0.0004" frictionloss="0.1" rgba="{rgba}" limited="true" range="{arm_range}">',
            f'      <site site="{b_site}"/>',
        ]
        for i in range(1, 13):
            lines.append(f'      <site site="{sites_prefix}s{i}_1"/>  <site site="{sites_prefix}s{i}_3"/>' if "_2" not in tname and "p2_t2" not in tname and "t2" not in tname else
                         f'      <site site="{sites_prefix}s{i}_2"/>  <site site="{sites_prefix}s{i}_4"/>')
        lines.append('    </spatial>')
        return "\n".join(lines)

    def gripper_tendon(tname, rgba, base_site, grip_site, rng):
        return f'''    <spatial name="{tname}" width="0.0004" frictionloss="0.1" rgba="{rgba}" limited="true" range="{rng}">
      <site site="{base_site}"/>
      <site site="{grip_site}"/>
    </spatial>'''

    # -----------------------------------------------------------------------
    # Full tendon block (manually faithful to original structure)
    # -----------------------------------------------------------------------
    def tendon_sites_arm(prefix, tendon_idx):
        """Return site list lines for arm tendon (1=odd sites, 2=even sites)."""
        lines = []
        if tendon_idx == 1:
            for i in range(1, 13):
                lines.append(f'      <site site="{prefix}s{i}_1"/>  <site site="{prefix}s{i}_3"/>')
        else:
            for i in range(1, 13):
                lines.append(f'      <site site="{prefix}s{i}_2"/>  <site site="{prefix}s{i}_4"/>')
        return "\n".join(lines)

    tendon_block = f"""  <tendon>
    <!-- P1 TDCR arm tendon 1 -->
    <spatial name="t1" width="0.0004" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{arm_range}">
      <site site="b_2"/>
{tendon_sites_arm("", 1)}
    </spatial>
    <!-- P1 TDCR arm tendon 2 -->
    <spatial name="t2" width="0.0004" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{arm_range}">
      <site site="b_1"/>
{tendon_sites_arm("", 2)}
    </spatial>
    <!-- P1 gripper tendons -->
    <spatial name="g1" width="0.0004" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{grp_range}">
      <site site="base_1"/>
      <site site="grip_1"/>
    </spatial>
    <spatial name="g2" width="0.0004" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{grp_range}">
      <site site="base_1"/>
      <site site="grip_2"/>
    </spatial>

    <!-- P2 TDCR arm tendon 1 -->
    <spatial name="p2_t1" width="0.0004" frictionloss="0.1" rgba=".3 .3 .95 1" limited="true" range="{arm_range}">
      <site site="p2_b_2"/>
{tendon_sites_arm("p2_", 1)}
    </spatial>
    <!-- P2 TDCR arm tendon 2 -->
    <spatial name="p2_t2" width="0.0004" frictionloss="0.1" rgba=".3 .3 .95 1" limited="true" range="{arm_range}">
      <site site="p2_b_1"/>
{tendon_sites_arm("p2_", 2)}
    </spatial>
    <!-- P2 gripper tendons -->
    <spatial name="p2_g1" width="0.0004" frictionloss="0.1" rgba=".3 .3 .95 1" limited="true" range="{grp_range}">
      <site site="p2_base_1"/>
      <site site="p2_grip_1"/>
    </spatial>
    <spatial name="p2_g2" width="0.0004" frictionloss="0.1" rgba=".3 .3 .95 1" limited="true" range="{grp_range}">
      <site site="p2_base_1"/>
      <site site="p2_grip_2"/>
    </spatial>
  </tendon>"""

    # -----------------------------------------------------------------------
    # Mesh asset block (scaled)
    # -----------------------------------------------------------------------
    def smesh(key):
        v = MESH_SCALES_ORIG[key] * S
        return f"{v:.9f} {v:.9f} {v:.9f}"

    asset_tdcr = f"""    <mesh name="platform_mesh"   file="3D/p_platform.STL"   scale="{smesh('platform_mesh')}"/>
    <mesh name="base_mesh"       file="3D/P_0Base.STL"       scale="{smesh('base_mesh')}"/>
    <mesh name="segment_mesh"    file="3D/P_1_3.STL"         scale="{smesh('segment_mesh')}"/>
    <mesh name="segment_2_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_2_mesh')}"/>
    <mesh name="segment_3_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_3_mesh')}"/>
    <mesh name="segment_4_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_4_mesh')}"/>
    <mesh name="segment_5_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_5_mesh')}"/>
    <mesh name="segment_6_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_6_mesh')}"/>
    <mesh name="segment_7_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_7_mesh')}"/>
    <mesh name="segment_8_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_8_mesh')}"/>
    <mesh name="segment_9_mesh"  file="3D/P_1_3.STL"         scale="{smesh('segment_9_mesh')}"/>
    <mesh name="segment_10_mesh" file="3D/P_1_3.STL"         scale="{smesh('segment_10_mesh')}"/>
    <mesh name="segment_11_mesh" file="3D/P_1_3.STL"         scale="{smesh('segment_11_mesh')}"/>
    <mesh name="segment_12_mesh" file="3D/P_1_3.STL"         scale="{smesh('segment_12_mesh')}"/>
    <mesh name="gripper_base"    file="3D_Gripper/gripper_base.STL" scale="{gms}"/>
    <mesh name="gripper"         file="3D_Gripper/gripper.STL"      scale="{gms}"/>"""

    # -----------------------------------------------------------------------
    # TDCR chain text for P1 and P2
    # -----------------------------------------------------------------------
    p1_tdcr = tdcr_chain("", "tdcr_")
    p2_tdcr = tdcr_chain("p2_", "p2_tdcr_")

    # -----------------------------------------------------------------------
    # Full XML
    # -----------------------------------------------------------------------
    xml = f"""<mujoco model="two_pandas_nohand_with_tdcr">
  <!--
  ╔══════════════════════════════════════════════════════╗
  ║  TDCR SCALE FACTOR = {S:<32.4g}         ║
  ║  Generated by generate_tdcr_xml.py                  ║
  ║  Panda arm geometry is NOT scaled.                  ║
  ╚══════════════════════════════════════════════════════╝
  -->
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option integrator="implicitfast" impratio="10" gravity="0 0 -9.81"/>
  <size nconmax="5000" njmax="5000"/>

  <statistic center="0 0 0.6" extent="1.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <!-- ═══════════════════════════════════════════════
       DEFAULTS
  ════════════════════════════════════════════════════ -->
  <default>
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
      <default class="panda/visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="panda/collision">
        <geom type="mesh" group="3"/>
      </default>
      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
  </default>

  <!-- ═══════════════════════════════════════════════
       ASSETS
  ════════════════════════════════════════════════════ -->
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="base_mat" rgba="0.25 0.25 0.28 1" reflectance="0.3"/>
    <material name="pole_mat" rgba="0.55 0.55 0.60 1" reflectance="0.5"/>

    <!-- panda materials -->
    <material class="panda" name="white"      rgba="1 1 1 1"/>
    <material class="panda" name="off_white"  rgba="0.901961 0.921569 0.929412 1"/>
    <material class="panda" name="dark_grey"  rgba="0.25 0.25 0.25 1"/>
    <material class="panda" name="green"      rgba="0 1 0 1"/>
    <material class="panda" name="light_blue" rgba="0.039216 0.541176 0.780392 1"/>

    <!-- panda collision meshes — UNSCALED -->
    <mesh name="link0_c"  file="link0.stl"/>
    <mesh name="link1_c"  file="link1.stl"/>
    <mesh name="link2_c"  file="link2.stl"/>
    <mesh name="link3_c"  file="link3.stl"/>
    <mesh name="link4_c"  file="link4.stl"/>
    <mesh name="link5_c0" file="link5_collision_0.obj"/>
    <mesh name="link5_c1" file="link5_collision_1.obj"/>
    <mesh name="link5_c2" file="link5_collision_2.obj"/>
    <mesh name="link6_c"  file="link6.stl"/>
    <mesh name="link7_c"  file="link7.stl"/>

    <!-- panda visual meshes — UNSCALED -->
    <mesh file="link0_0.obj"/>  <mesh file="link0_1.obj"/>  <mesh file="link0_2.obj"/>
    <mesh file="link0_3.obj"/>  <mesh file="link0_4.obj"/>  <mesh file="link0_5.obj"/>
    <mesh file="link0_7.obj"/>  <mesh file="link0_8.obj"/>  <mesh file="link0_9.obj"/>
    <mesh file="link0_10.obj"/> <mesh file="link0_11.obj"/>
    <mesh file="link1.obj"/>    <mesh file="link2.obj"/>
    <mesh file="link3_0.obj"/>  <mesh file="link3_1.obj"/>  <mesh file="link3_2.obj"/>  <mesh file="link3_3.obj"/>
    <mesh file="link4_0.obj"/>  <mesh file="link4_1.obj"/>  <mesh file="link4_2.obj"/>  <mesh file="link4_3.obj"/>
    <mesh file="link5_0.obj"/>  <mesh file="link5_1.obj"/>  <mesh file="link5_2.obj"/>
    <mesh file="link6_0.obj"/>  <mesh file="link6_1.obj"/>  <mesh file="link6_2.obj"/>  <mesh file="link6_3.obj"/>
    <mesh file="link6_4.obj"/>  <mesh file="link6_5.obj"/>  <mesh file="link6_6.obj"/>  <mesh file="link6_7.obj"/>
    <mesh file="link6_8.obj"/>  <mesh file="link6_9.obj"/>  <mesh file="link6_10.obj"/> <mesh file="link6_11.obj"/>
    <mesh file="link6_12.obj"/> <mesh file="link6_13.obj"/> <mesh file="link6_14.obj"/> <mesh file="link6_15.obj"/>
    <mesh file="link6_16.obj"/>
    <mesh file="link7_0.obj"/>  <mesh file="link7_1.obj"/>  <mesh file="link7_2.obj"/>  <mesh file="link7_3.obj"/>
    <mesh file="link7_4.obj"/>  <mesh file="link7_5.obj"/>  <mesh file="link7_6.obj"/>  <mesh file="link7_7.obj"/>

    <!-- TDCR meshes — SCALED by {S} -->
{asset_tdcr}
  </asset>

  <!-- ═══════════════════════════════════════════════
       WORLD
  ════════════════════════════════════════════════════ -->
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <light pos="-1 0 2" dir="0.5 0 -1"  directional="false" diffuse="0.4 0.4 0.4"/>
    <light pos=" 1 0 2" dir="-0.5 0 -1" directional="false" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- ── Cuboid base platform ── -->
    <geom name="base_platform" type="box"
          size="0.85 0.3 0.1" pos="0 0 0.1" material="base_mat"/>
    <geom name="centre_pole" type="cylinder"
          fromto="0 0 0.2  0 0 0.9" size="0.04" material="pole_mat"/>
    <geom name="pole_cap" type="sphere"
          size="0.055" pos="0 0 0.92" material="pole_mat"/>

    <!-- ══════════════════════════════════════════════════════
         PANDA 1  –  left (x=-0.55), facing +Y  — PANDA UNSCALED
    ═══════════════════════════════════════════════════════════ -->
    <body name="p1_link0" childclass="panda" pos="-0.55 0 0.2" euler="0 0 1.5708">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
      <geom mesh="link0_0"  material="off_white"  class="panda/visual"/>
      <geom mesh="link0_1"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_2"  material="off_white"  class="panda/visual"/>
      <geom mesh="link0_3"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_4"  material="off_white"  class="panda/visual"/>
      <geom mesh="link0_5"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_7"  material="white"      class="panda/visual"/>
      <geom mesh="link0_8"  material="white"      class="panda/visual"/>
      <geom mesh="link0_9"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_10" material="off_white"  class="panda/visual"/>
      <geom mesh="link0_11" material="white"      class="panda/visual"/>
      <geom mesh="link0_c"  class="panda/collision"/>
      <body name="p1_link1" pos="0 0 0.333">
        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762"
          fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
        <joint name="p1_joint1"/>
        <geom material="white" mesh="link1" class="panda/visual"/>
        <geom mesh="link1_c" class="panda/collision"/>
        <body name="p1_link2" quat="1 -1 0 0">
          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495"
            fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
          <joint name="p1_joint2" range="-1.7628 1.7628"/>
          <geom material="white" mesh="link2" class="panda/visual"/>
          <geom mesh="link2_c" class="panda/collision"/>
          <body name="p1_link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="p1_joint3"/>
            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
              fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
            <geom mesh="link3_0" material="white"     class="panda/visual"/>
            <geom mesh="link3_1" material="white"     class="panda/visual"/>
            <geom mesh="link3_2" material="white"     class="panda/visual"/>
            <geom mesh="link3_3" material="dark_grey" class="panda/visual"/>
            <geom mesh="link3_c" class="panda/collision"/>
            <body name="p1_link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
              <joint name="p1_joint4" range="-3.0718 -0.0698"/>
              <geom mesh="link4_0" material="white"     class="panda/visual"/>
              <geom mesh="link4_1" material="white"     class="panda/visual"/>
              <geom mesh="link4_2" material="dark_grey" class="panda/visual"/>
              <geom mesh="link4_3" material="white"     class="panda/visual"/>
              <geom mesh="link4_c" class="panda/collision"/>
              <body name="p1_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                  fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                <joint name="p1_joint5"/>
                <geom mesh="link5_0" material="dark_grey" class="panda/visual"/>
                <geom mesh="link5_1" material="white"     class="panda/visual"/>
                <geom mesh="link5_2" material="white"     class="panda/visual"/>
                <geom mesh="link5_c0" class="panda/collision"/>
                <geom mesh="link5_c1" class="panda/collision"/>
                <geom mesh="link5_c2" class="panda/collision"/>
                <body name="p1_link6" quat="1 1 0 0">
                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                    fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                  <joint name="p1_joint6" range="-0.0175 3.7525"/>
                  <geom mesh="link6_0"  material="off_white"  class="panda/visual"/>
                  <geom mesh="link6_1"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_2"  material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_3"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_4"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_5"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_6"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_7"  material="light_blue" class="panda/visual"/>
                  <geom mesh="link6_8"  material="light_blue" class="panda/visual"/>
                  <geom mesh="link6_9"  material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_10" material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_11" material="white"      class="panda/visual"/>
                  <geom mesh="link6_12" material="green"      class="panda/visual"/>
                  <geom mesh="link6_13" material="white"      class="panda/visual"/>
                  <geom mesh="link6_14" material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_15" material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_16" material="white"      class="panda/visual"/>
                  <geom mesh="link6_c" class="panda/collision"/>
                  <body name="p1_link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2"
                      fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                    <joint name="p1_joint7"/>
                    <geom mesh="link7_0" material="white"     class="panda/visual"/>
                    <geom mesh="link7_1" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_2" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_3" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_4" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_5" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_6" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_7" material="white"     class="panda/visual"/>
                    <geom mesh="link7_c" class="panda/collision"/>

                    <!-- P1 TDCR (scale={S}) -->
{p1_tdcr}
                  </body><!-- p1_link7 -->
                </body><!-- p1_link6 -->
              </body><!-- p1_link5 -->
            </body><!-- p1_link4 -->
          </body><!-- p1_link3 -->
        </body><!-- p1_link2 -->
      </body><!-- p1_link1 -->
    </body><!-- p1_link0 -->

    <!-- ══════════════════════════════════════════════════════
         PANDA 2  –  right (x=+0.55), facing +Y — PANDA UNSCALED
    ═══════════════════════════════════════════════════════════ -->
    <body name="p2_link0" childclass="panda" pos="0.55 0 0.2" euler="0 0 1.5708">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
      <geom mesh="link0_0"  material="off_white"  class="panda/visual"/>
      <geom mesh="link0_1"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_2"  material="off_white"  class="panda/visual"/>
      <geom mesh="link0_3"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_4"  material="off_white"  class="panda/visual"/>
      <geom mesh="link0_5"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_7"  material="white"      class="panda/visual"/>
      <geom mesh="link0_8"  material="white"      class="panda/visual"/>
      <geom mesh="link0_9"  material="dark_grey"  class="panda/visual"/>
      <geom mesh="link0_10" material="off_white"  class="panda/visual"/>
      <geom mesh="link0_11" material="white"      class="panda/visual"/>
      <geom mesh="link0_c"  class="panda/collision"/>
      <body name="p2_link1" pos="0 0 0.333">
        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762"
          fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
        <joint name="p2_joint1"/>
        <geom material="white" mesh="link1" class="panda/visual"/>
        <geom mesh="link1_c" class="panda/collision"/>
        <body name="p2_link2" quat="1 -1 0 0">
          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495"
            fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
          <joint name="p2_joint2" range="-1.7628 1.7628"/>
          <geom material="white" mesh="link2" class="panda/visual"/>
          <geom mesh="link2_c" class="panda/collision"/>
          <body name="p2_link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="p2_joint3"/>
            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
              fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
            <geom mesh="link3_0" material="white"     class="panda/visual"/>
            <geom mesh="link3_1" material="white"     class="panda/visual"/>
            <geom mesh="link3_2" material="white"     class="panda/visual"/>
            <geom mesh="link3_3" material="dark_grey" class="panda/visual"/>
            <geom mesh="link3_c" class="panda/collision"/>
            <body name="p2_link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
              <joint name="p2_joint4" range="-3.0718 -0.0698"/>
              <geom mesh="link4_0" material="white"     class="panda/visual"/>
              <geom mesh="link4_1" material="white"     class="panda/visual"/>
              <geom mesh="link4_2" material="dark_grey" class="panda/visual"/>
              <geom mesh="link4_3" material="white"     class="panda/visual"/>
              <geom mesh="link4_c" class="panda/collision"/>
              <body name="p2_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                  fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                <joint name="p2_joint5"/>
                <geom mesh="link5_0" material="dark_grey" class="panda/visual"/>
                <geom mesh="link5_1" material="white"     class="panda/visual"/>
                <geom mesh="link5_2" material="white"     class="panda/visual"/>
                <geom mesh="link5_c0" class="panda/collision"/>
                <geom mesh="link5_c1" class="panda/collision"/>
                <geom mesh="link5_c2" class="panda/collision"/>
                <body name="p2_link6" quat="1 1 0 0">
                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                    fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                  <joint name="p2_joint6" range="-0.0175 3.7525"/>
                  <geom mesh="link6_0"  material="off_white"  class="panda/visual"/>
                  <geom mesh="link6_1"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_2"  material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_3"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_4"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_5"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_6"  material="white"      class="panda/visual"/>
                  <geom mesh="link6_7"  material="light_blue" class="panda/visual"/>
                  <geom mesh="link6_8"  material="light_blue" class="panda/visual"/>
                  <geom mesh="link6_9"  material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_10" material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_11" material="white"      class="panda/visual"/>
                  <geom mesh="link6_12" material="green"      class="panda/visual"/>
                  <geom mesh="link6_13" material="white"      class="panda/visual"/>
                  <geom mesh="link6_14" material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_15" material="dark_grey"  class="panda/visual"/>
                  <geom mesh="link6_16" material="white"      class="panda/visual"/>
                  <geom mesh="link6_c" class="panda/collision"/>
                  <body name="p2_link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2"
                      fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                    <joint name="p2_joint7"/>
                    <geom mesh="link7_0" material="white"     class="panda/visual"/>
                    <geom mesh="link7_1" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_2" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_3" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_4" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_5" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_6" material="dark_grey" class="panda/visual"/>
                    <geom mesh="link7_7" material="white"     class="panda/visual"/>
                    <geom mesh="link7_c" class="panda/collision"/>

                    <!-- P2 TDCR (scale={S}) -->
{p2_tdcr}
                  </body><!-- p2_link7 -->
                </body><!-- p2_link6 -->
              </body><!-- p2_link5 -->
            </body><!-- p2_link4 -->
          </body><!-- p2_link3 -->
        </body><!-- p2_link2 -->
      </body><!-- p2_link1 -->
    </body><!-- p2_link0 -->

  </worldbody>

  <!-- ═══════════════════════════════════════════════
       TENDONS
       Tendon ranges scaled by {S}:
         arm:     {ARM_TENDON_RANGE_ORIG[0]}–{ARM_TENDON_RANGE_ORIG[1]} → {f(ARM_TENDON_RANGE_ORIG[0]*S)}–{f(ARM_TENDON_RANGE_ORIG[1]*S)}
         gripper: {GRIPPER_TENDON_RANGE_ORIG[0]}–{GRIPPER_TENDON_RANGE_ORIG[1]} → {f(GRIPPER_TENDON_RANGE_ORIG[0]*S)}–{f(GRIPPER_TENDON_RANGE_ORIG[1]*S)}
  ════════════════════════════════════════════════════ -->
{tendon_block}

  <!-- ═══════════════════════════════════════════════
       ACTUATORS
       Muscle lengthrange scaled by {S}:
         arm:     {ARM_TENDON_RANGE_ORIG[0]}–{ARM_TENDON_RANGE_ORIG[1]} → {f(ARM_TENDON_RANGE_ORIG[0]*S)}–{f(ARM_TENDON_RANGE_ORIG[1]*S)}
         gripper: {GRIPPER_TENDON_RANGE_ORIG[0]}–{GRIPPER_TENDON_RANGE_ORIG[1]} → {f(GRIPPER_TENDON_RANGE_ORIG[0]*S)}–{f(GRIPPER_TENDON_RANGE_ORIG[1]*S)}
  ════════════════════════════════════════════════════ -->
  <actuator>
    <!-- Panda 1 — UNSCALED -->
    <general class="panda" name="p1_actuator1" joint="p1_joint1" gainprm="4500" biasprm="0 -4500 -450"/>
    <general class="panda" name="p1_actuator2" joint="p1_joint2" gainprm="4500" biasprm="0 -4500 -450" ctrlrange="-1.7628 1.7628"/>
    <general class="panda" name="p1_actuator3" joint="p1_joint3" gainprm="3500" biasprm="0 -3500 -350"/>
    <general class="panda" name="p1_actuator4" joint="p1_joint4" gainprm="3500" biasprm="0 -3500 -350" ctrlrange="-3.0718 -0.0698"/>
    <general class="panda" name="p1_actuator5" joint="p1_joint5" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <general class="panda" name="p1_actuator6" joint="p1_joint6" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12" ctrlrange="-0.0175 3.7525"/>
    <general class="panda" name="p1_actuator7" joint="p1_joint7" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <!-- Panda 2 — UNSCALED -->
    <general class="panda" name="p2_actuator1" joint="p2_joint1" gainprm="4500" biasprm="0 -4500 -450"/>
    <general class="panda" name="p2_actuator2" joint="p2_joint2" gainprm="4500" biasprm="0 -4500 -450" ctrlrange="-1.7628 1.7628"/>
    <general class="panda" name="p2_actuator3" joint="p2_joint3" gainprm="3500" biasprm="0 -3500 -350"/>
    <general class="panda" name="p2_actuator4" joint="p2_joint4" gainprm="3500" biasprm="0 -3500 -350" ctrlrange="-3.0718 -0.0698"/>
    <general class="panda" name="p2_actuator5" joint="p2_joint5" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <general class="panda" name="p2_actuator6" joint="p2_joint6" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12" ctrlrange="-0.0175 3.7525"/>
    <general class="panda" name="p2_actuator7" joint="p2_joint7" gainprm="2000" biasprm="0 -2000 -200" forcerange="-12 12"/>
    <!-- P1 TDCR muscles — lengthrange scaled -->
    <muscle name="A_1" ctrllimited="true" lengthrange="{arm_range}"  ctrlrange="0 1" force="2"     tendon="t1"/>
    <muscle name="A_2" ctrllimited="true" lengthrange="{arm_range}"  ctrlrange="0 1" force="2"     tendon="t2"/>
    <muscle name="A_3" ctrllimited="true" lengthrange="{grp_range}" ctrlrange="0 1" force="300.5" tendon="g1"/>
    <muscle name="A_4" ctrllimited="true" lengthrange="{grp_range}" ctrlrange="0 1" force="300.5" tendon="g2"/>
    <!-- P2 TDCR muscles — lengthrange scaled -->
    <muscle name="B_1" ctrllimited="true" lengthrange="{arm_range}"  ctrlrange="0 1" force="2"     tendon="p2_t1"/>
    <muscle name="B_2" ctrllimited="true" lengthrange="{arm_range}"  ctrlrange="0 1" force="2"     tendon="p2_t2"/>
    <muscle name="B_3" ctrllimited="true" lengthrange="{grp_range}" ctrlrange="0 1" force="300.5" tendon="p2_g1"/>
    <muscle name="B_4" ctrllimited="true" lengthrange="{grp_range}" ctrlrange="0 1" force="300.5" tendon="p2_g2"/>
  </actuator>

  <!-- ═══════════════════════════════════════════════
       CONTACT EXCLUSIONS
  ════════════════════════════════════════════════════ -->
  <contact>
    <exclude body1="p1_link0" body2="p1_link1"/>
    <exclude body1="p2_link0" body2="p2_link1"/>
  </contact>

  <!-- ═══════════════════════════════════════════════
       KEYFRAME
       qpos: 7 (p1 arm) + 7 (p2 arm)
             + 12 (P1 TDCR) + 2 (P1 gripper)
             + 12 (P2 TDCR) + 2 (P2 gripper) = 42
       ctrl: 7 (p1) + 7 (p2) + 4 (P1 TDCR) + 4 (P2 TDCR) = 22
  ════════════════════════════════════════════════════ -->
  <keyframe>
    <key name="home"
      qpos="0 0 0 -1.57079 0 1.57079 -0.7853
            0 0 0 -1.57079 0 1.57079 -0.7853
            0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0"
      ctrl="0 0 0 -1.57079 0 1.57079 -0.7853
            0 0 0 -1.57079 0 1.57079 -0.7853
            0 0 0 0
            0 0 0 0"/>
  </keyframe>

</mujoco>
"""
    return xml


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo XML for two-Panda + TDCR system with a given TDCR scale factor."
    )
    parser.add_argument(
        "scale", nargs="?", type=float, default=None,
        help="TDCR scale factor (e.g. 0.6, 1.0, 1.5)"
    )
    parser.add_argument(
        "--scale", dest="scale_flag", type=float, default=None,
        help="TDCR scale factor (alternative flag form)"
    )
    parser.add_argument(
        "--out", "-o", type=str, default=None,
        help="Output file path (default: two_pandas_tdcr_scale<SCALE>.xml)"
    )
    args = parser.parse_args()

    scale = args.scale if args.scale is not None else args.scale_flag

    if scale is None:
        try:
            scale = float(0.15)
        except (ValueError, EOFError):
            print("Invalid scale factor. Exiting.", file=sys.stderr)
            sys.exit(1)

    if scale <= 0:
        print("Scale factor must be positive.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or f"two_pandas_tdcr_scale{scale:.4g}.xml"

    print(f"Generating XML with TDCR scale = {scale} ...")
    xml = build_xml(scale)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(xml)

    print(f"Written to: {out_path}  ({len(xml):,} characters)")


if __name__ == "__main__":
    main()