# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: telop_vr.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='telop_vr.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\x0etelop_vr.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"\x97\x01\n\x0eTelopVRSession\x12\x33\n\x0ftimestamp_start\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x31\n\rtimestamp_end\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x1d\n\x06states\x18\x03 \x03(\x0b\x32\r.TelopVRState\"\xa2\x01\n\x0cTelopVRState\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\"\n\x08snapshot\x18\x02 \x01(\x0b\x32\x10.TelopVRSnapshot\x12!\n\x08grippers\x18\x03 \x03(\x0b\x32\x0f.TelopVRGripper\x12\x1c\n\x05items\x18\x04 \x03(\x0b\x32\r.TelopVRItems\"~\n\x0fTelopVRSnapshot\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04rows\x18\x02 \x01(\r\x12\x0f\n\x07\x63olumns\x18\x03 \x01(\r\x12\x0e\n\x06\x64\x61ta_r\x18\x04 \x03(\r\x12\x0e\n\x06\x64\x61ta_g\x18\x05 \x03(\r\x12\x0e\n\x06\x64\x61ta_b\x18\x06 \x03(\r\x12\x0e\n\x06\x64\x61ta_d\x18\x07 \x03(\r\"\xb6\x01\n\x0eTelopVRGripper\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\x0c\n\x04roll\x18\x04 \x01(\x02\x12\r\n\x05pitch\x18\x05 \x01(\x02\x12\x0b\n\x03yaw\x18\x06 \x01(\x02\x12\n\n\x02\x64x\x18\x07 \x01(\x02\x12\n\n\x02\x64y\x18\x08 \x01(\x02\x12\n\n\x02\x64z\x18\t \x01(\x02\x12\x0b\n\x03\x64rx\x18\n \x01(\x02\x12\x0b\n\x03\x64ry\x18\x0b \x01(\x02\x12\x0b\n\x03\x64rz\x18\x0c \x01(\x02\x12\x0e\n\x06\x66inger\x18\r \x01(\x02\"e\n\x0cTelopVRItems\x12\n\n\x02id\x18\x01 \x01(\r\x12\t\n\x01x\x18\x02 \x01(\x02\x12\t\n\x01y\x18\x03 \x01(\x02\x12\t\n\x01z\x18\x04 \x01(\x02\x12\x0c\n\x04roll\x18\x05 \x01(\x02\x12\r\n\x05pitch\x18\x06 \x01(\x02\x12\x0b\n\x03yaw\x18\x07 \x01(\x02\x62\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,])




_TELOPVRSESSION = _descriptor.Descriptor(
  name='TelopVRSession',
  full_name='TelopVRSession',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp_start', full_name='TelopVRSession.timestamp_start', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='timestamp_end', full_name='TelopVRSession.timestamp_end', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='states', full_name='TelopVRSession.states', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=52,
  serialized_end=203,
)


_TELOPVRSTATE = _descriptor.Descriptor(
  name='TelopVRState',
  full_name='TelopVRState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='TelopVRState.timestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='snapshot', full_name='TelopVRState.snapshot', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='grippers', full_name='TelopVRState.grippers', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='items', full_name='TelopVRState.items', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=206,
  serialized_end=368,
)


_TELOPVRSNAPSHOT = _descriptor.Descriptor(
  name='TelopVRSnapshot',
  full_name='TelopVRSnapshot',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='TelopVRSnapshot.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rows', full_name='TelopVRSnapshot.rows', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='columns', full_name='TelopVRSnapshot.columns', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_r', full_name='TelopVRSnapshot.data_r', index=3,
      number=4, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_g', full_name='TelopVRSnapshot.data_g', index=4,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_b', full_name='TelopVRSnapshot.data_b', index=5,
      number=6, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_d', full_name='TelopVRSnapshot.data_d', index=6,
      number=7, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=370,
  serialized_end=496,
)


_TELOPVRGRIPPER = _descriptor.Descriptor(
  name='TelopVRGripper',
  full_name='TelopVRGripper',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='TelopVRGripper.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='TelopVRGripper.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='TelopVRGripper.z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roll', full_name='TelopVRGripper.roll', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pitch', full_name='TelopVRGripper.pitch', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='yaw', full_name='TelopVRGripper.yaw', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dx', full_name='TelopVRGripper.dx', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dy', full_name='TelopVRGripper.dy', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dz', full_name='TelopVRGripper.dz', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='drx', full_name='TelopVRGripper.drx', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dry', full_name='TelopVRGripper.dry', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='drz', full_name='TelopVRGripper.drz', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='finger', full_name='TelopVRGripper.finger', index=12,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=499,
  serialized_end=681,
)


_TELOPVRITEMS = _descriptor.Descriptor(
  name='TelopVRItems',
  full_name='TelopVRItems',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='TelopVRItems.id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='x', full_name='TelopVRItems.x', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='TelopVRItems.y', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='TelopVRItems.z', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roll', full_name='TelopVRItems.roll', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pitch', full_name='TelopVRItems.pitch', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='yaw', full_name='TelopVRItems.yaw', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=683,
  serialized_end=784,
)

_TELOPVRSESSION.fields_by_name['timestamp_start'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_TELOPVRSESSION.fields_by_name['timestamp_end'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_TELOPVRSESSION.fields_by_name['states'].message_type = _TELOPVRSTATE
_TELOPVRSTATE.fields_by_name['timestamp'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_TELOPVRSTATE.fields_by_name['snapshot'].message_type = _TELOPVRSNAPSHOT
_TELOPVRSTATE.fields_by_name['grippers'].message_type = _TELOPVRGRIPPER
_TELOPVRSTATE.fields_by_name['items'].message_type = _TELOPVRITEMS
DESCRIPTOR.message_types_by_name['TelopVRSession'] = _TELOPVRSESSION
DESCRIPTOR.message_types_by_name['TelopVRState'] = _TELOPVRSTATE
DESCRIPTOR.message_types_by_name['TelopVRSnapshot'] = _TELOPVRSNAPSHOT
DESCRIPTOR.message_types_by_name['TelopVRGripper'] = _TELOPVRGRIPPER
DESCRIPTOR.message_types_by_name['TelopVRItems'] = _TELOPVRITEMS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TelopVRSession = _reflection.GeneratedProtocolMessageType('TelopVRSession', (_message.Message,), dict(
  DESCRIPTOR = _TELOPVRSESSION,
  __module__ = 'telop_vr_pb2'
  # @@protoc_insertion_point(class_scope:TelopVRSession)
  ))
_sym_db.RegisterMessage(TelopVRSession)

TelopVRState = _reflection.GeneratedProtocolMessageType('TelopVRState', (_message.Message,), dict(
  DESCRIPTOR = _TELOPVRSTATE,
  __module__ = 'telop_vr_pb2'
  # @@protoc_insertion_point(class_scope:TelopVRState)
  ))
_sym_db.RegisterMessage(TelopVRState)

TelopVRSnapshot = _reflection.GeneratedProtocolMessageType('TelopVRSnapshot', (_message.Message,), dict(
  DESCRIPTOR = _TELOPVRSNAPSHOT,
  __module__ = 'telop_vr_pb2'
  # @@protoc_insertion_point(class_scope:TelopVRSnapshot)
  ))
_sym_db.RegisterMessage(TelopVRSnapshot)

TelopVRGripper = _reflection.GeneratedProtocolMessageType('TelopVRGripper', (_message.Message,), dict(
  DESCRIPTOR = _TELOPVRGRIPPER,
  __module__ = 'telop_vr_pb2'
  # @@protoc_insertion_point(class_scope:TelopVRGripper)
  ))
_sym_db.RegisterMessage(TelopVRGripper)

TelopVRItems = _reflection.GeneratedProtocolMessageType('TelopVRItems', (_message.Message,), dict(
  DESCRIPTOR = _TELOPVRITEMS,
  __module__ = 'telop_vr_pb2'
  # @@protoc_insertion_point(class_scope:TelopVRItems)
  ))
_sym_db.RegisterMessage(TelopVRItems)


# @@protoc_insertion_point(module_scope)
